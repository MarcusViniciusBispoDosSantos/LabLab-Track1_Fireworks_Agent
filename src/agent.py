from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .classifier import TaskType, classify_task
from .models import best_model, ranked_models, secondary_models, is_reasoning_model
from .prompts import system_for
from .solvers import try_local

@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str
    allowed_models: list[str]
    timeout: float = 35.0
    retries: int = 3
    local_fast_paths: bool = True
    max_api_tasks: int = 999
    hard_second_try: bool = False

    @classmethod
    def from_env(cls) -> 'AgentConfig':
        missing = [k for k in ('FIREWORKS_API_KEY','FIREWORKS_BASE_URL','ALLOWED_MODELS') if not os.getenv(k)]
        if missing:
            raise RuntimeError('Missing required environment variable(s): ' + ', '.join(missing))
        models = [m.strip() for m in os.environ['ALLOWED_MODELS'].replace('\n', ',').split(',') if m.strip()]
        return cls(
            api_key=os.environ['FIREWORKS_API_KEY'],
            base_url=os.environ['FIREWORKS_BASE_URL'].rstrip('/'),
            allowed_models=models,
            timeout=float(os.getenv('REQUEST_TIMEOUT_SECONDS','35')),
            retries=max(1, int(os.getenv('MAX_RETRIES','3'))),
            local_fast_paths=os.getenv('ENABLE_LOCAL_FAST_PATHS','1').lower() in {'1','true','yes','on'},
            max_api_tasks=max(1, int(os.getenv('MAX_API_TASKS','999'))),
            hard_second_try=os.getenv('HARD_SECOND_TRY','0').lower() in {'1','true','yes','on'},
        )

class FireworksTrack1Agent:
    '''v13 token-aware accuracy agent.

    Goal: recover accuracy while keeping proxy tokens low. It solves very high-confidence
    math/sentiment/template-code/assignment-logic locally, then makes exactly one concise
    Fireworks call for remaining tasks. Optional HARD_SECOND_TRY is disabled by default to
    stay near the user's <1400-token target.
    '''
    def __init__(self, config: AgentConfig):
        self.config = config
        self.api_calls = 0

    def answer_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [self.answer_task(t) for t in tasks]

    def answer_task(self, task: dict[str, Any]) -> dict[str, str]:
        tid = str(task.get('task_id','')) or 'missing_task_id'
        prompt = str(task.get('prompt','')).strip()
        if not prompt:
            return {'task_id': tid, 'answer': ''}
        typ = classify_task(prompt)

        # Safe local exact solvers use zero proxy tokens and are restricted to high-confidence patterns.
        if self.config.local_fast_paths:
            local = try_local(prompt, typ)
            if local and _safe_to_use_local(local, typ, prompt):
                return {'task_id': tid, 'answer': cleanup(local, typ, prompt)}

        if self.api_calls >= self.config.max_api_tasks:
            return {'task_id': tid, 'answer': fallback_answer(prompt, typ)}

        try:
            ans = self.direct(prompt, typ, best_model(self.config.allowed_models, typ))
            ans = cleanup(ans, typ, prompt)
            if self.config.hard_second_try and typ in {TaskType.MATH, TaskType.LOGIC, TaskType.CODE_DEBUG, TaskType.CODE_GEN} and needs_fix(ans, typ, prompt):
                for model in secondary_models(self.config.allowed_models, typ, 2)[1:]:
                    if self.api_calls >= self.config.max_api_tasks: break
                    ans2 = cleanup(self.direct(prompt, typ, model), typ, prompt)
                    if ans2 and not needs_fix(ans2, typ, prompt):
                        ans = ans2; break
            if not ans:
                ans = fallback_answer(prompt, typ)
            return {'task_id': tid, 'answer': ans}
        except Exception:
            return {'task_id': tid, 'answer': fallback_answer(prompt, typ)}

    def direct(self, prompt: str, typ: TaskType, model: str) -> str:
        self.api_calls += 1
        messages = [
            {'role': 'system', 'content': system_for(typ)},
            {'role': 'user', 'content': add_reasoning_guard(prompt, model)},
        ]
        payload = {'model': model, 'messages': messages, 'temperature': 0, 'top_p': 1, 'max_tokens': max_tokens(typ)}
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        url = self.config.base_url if self.config.base_url.endswith('/chat/completions') else self.config.base_url + '/chat/completions'
        last = None
        for i in range(self.config.retries):
            req = urllib.request.Request(url, data=body, method='POST', headers={
                'Authorization': 'Bearer ' + self.config.api_key,
                'Content-Type': 'application/json', 'Accept': 'application/json'
            })
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as r:
                    data = json.loads(r.read().decode('utf-8'))
                return extract_content(data)
            except urllib.error.HTTPError as e:
                last = f'HTTP {e.code}: {e.read().decode("utf-8", errors="replace")[:400]}'
                if e.code not in {408,409,425,429,500,502,503,504}: break
            except Exception as e:
                last = str(e)
            time.sleep(min(2.0, 0.4 * (2 ** i)))
        raise RuntimeError(last or 'request failed')

def add_reasoning_guard(prompt: str, model: str) -> str:
    if is_reasoning_model(model):
        return prompt + '\n\nReturn only the final answer. Do not include chain-of-thought or <think> text.'
    return prompt

def extract_content(data: dict[str, Any]) -> str:
    choices = data.get('choices') or []
    if not choices: raise RuntimeError('no choices')
    c = choices[0]
    msg = c.get('message') if isinstance(c, dict) else None
    if isinstance(msg, dict):
        content = msg.get('content')
        if isinstance(content, str): return remove_thinking(content).strip()
        if isinstance(content, list):
            parts=[]
            for x in content:
                if isinstance(x, dict):
                    t=x.get('text') or x.get('content')
                    if isinstance(t, str): parts.append(t)
                elif isinstance(x, str): parts.append(x)
            return remove_thinking('\n'.join(parts)).strip()
    if isinstance(c, dict) and isinstance(c.get('text'), str): return remove_thinking(c['text']).strip()
    raise RuntimeError('bad response')

def max_tokens(t: TaskType) -> int:
    # Intentionally small to target <1400 proxy tokens. Increase CODE_GEN only when needed.
    return {
        TaskType.FACTUAL: 120,
        TaskType.MATH: 180,
        TaskType.SENTIMENT: 60,
        TaskType.SUMMARY: 140,
        TaskType.NER: 170,
        TaskType.CODE_DEBUG: 420,
        TaskType.LOGIC: 260,
        TaskType.CODE_GEN: 520,
    }[t]

def remove_thinking(text: str) -> str:
    s = (text or '').strip()
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.I|re.S).strip()
    if '</think>' in s.lower(): s = re.split(r'</think>', s, flags=re.I)[-1].strip()
    m = re.fullmatch(r'(?:final\s+answer|answer|result)\s*:\s*(.+)', s, flags=re.I|re.S)
    return m.group(1).strip() if m else s

def cleanup(ans: str, typ: TaskType, prompt: str) -> str:
    s = remove_thinking(ans).strip()
    s = re.sub(r'^```(?:\w+)?\s*', '', s).strip()
    s = re.sub(r'\s*```$', '', s).strip()
    s = re.sub(r'^(?:Sure|Of course)[,.!]?\s*', '', s, flags=re.I).strip()
    lowp = prompt.lower()
    if typ == TaskType.SENTIMENT:
        lab = sentiment_label(s, prompt)
        if lab: return lab
    if 'json' in lowp:
        j = extract_json(s)
        if j: return j
    if re.search(r'\b(answer yes or no|yes/no)\b', lowp):
        yn = re.search(r'\b(yes|no)\b', s, re.I)
        if yn: return yn.group(1).capitalize()
    if re.search(r'\b(true or false|true/false)\b', lowp):
        tf = re.search(r'\b(true|false)\b', s, re.I)
        if tf: return tf.group(1).capitalize()
    if 'final number only' in lowp or 'answer only' in lowp or 'final answer only' in lowp:
        line = [x.strip() for x in s.splitlines() if x.strip()]
        if line: return line[-1]
    return s

def sentiment_label(text: str, prompt: str) -> str | None:
    labels = requested_labels(prompt) or ['positive','negative','neutral','mixed']
    # Prefer first valid label mentioned in answer.
    for lab in labels:
        if re.search(rf'\b{re.escape(lab)}\b', text, re.I):
            return lab.capitalize()
    return None

def requested_labels(prompt: str) -> list[str] | None:
    low = prompt.lower()
    known = [x for x in ['positive','negative','neutral','mixed'] if re.search(rf'\b{x}\b', low)]
    return known if len(known) >= 2 else None

def extract_json(s: str) -> str | None:
    for start, end in [('{','}'), ('[',']')]:
        a, b = s.find(start), s.rfind(end)
        if a != -1 and b > a:
            cand = s[a:b+1]
            try:
                json.loads(cand); return cand
            except Exception: pass
    return None

def needs_fix(ans: str, typ: TaskType, prompt: str) -> bool:
    if not ans: return True
    low = ans.lower()
    if any(x in low for x in ['traceback', 'cannot answer', 'i don\'t know', 'error:']): return True
    if typ == TaskType.MATH and not re.search(r'[-+]?\d', ans): return True
    if typ == TaskType.SENTIMENT and not sentiment_label(ans, prompt): return True
    if typ == TaskType.CODE_GEN and ('function' in prompt.lower() or 'python' in prompt.lower()) and not re.search(r'\bdef\s+\w+\s*\(', ans): return True
    return False

def fallback_answer(prompt: str, typ: TaskType) -> str:
    # This should rarely be used; keep it valid and concise.
    if typ == TaskType.SENTIMENT: return 'Neutral'
    if typ == TaskType.NER: return 'No named entities found.'
    return 'I don\'t know.'

def _safe_to_use_local(ans: str, typ: TaskType, prompt: str) -> bool:
    if not ans: return False
    lowp = prompt.lower()
    if any(x in lowp for x in ['exactly', 'json', 'xml', 'yaml', 'table']) and typ not in {TaskType.MATH, TaskType.SENTIMENT}:
        return False
    if typ == TaskType.MATH: return bool(re.search(r'[-+]?\d', ans)) and len(ans) < 80
    if typ == TaskType.SENTIMENT: return bool(sentiment_label(ans, prompt))
    if typ in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}: return 'def ' in ans or 'Bug:' in ans
    if typ == TaskType.LOGIC: return '=' in ans and len(ans) < 300
    return False
