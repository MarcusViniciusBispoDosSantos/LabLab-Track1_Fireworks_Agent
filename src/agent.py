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
from .models import is_thinking_model, parse_allowed_models, primary_models, ranked_models, selector_model, usable_models
from .prompts import REPAIR_SYSTEM, SELECTOR_SYSTEM, system_for
from .solvers import try_solve_locally


BATCH_SYSTEM = """You are an expert hidden-benchmark solver for a general-purpose AI agent.
You will receive a JSON array of tasks. Each task has task_id and prompt.
Return ONLY a valid JSON array. No markdown. No explanations outside JSON.
Each output item must be exactly: {"task_id":"same id","answer":"final answer string"}.
Solve every task independently and obey the exact format requested inside each prompt.
Think privately. Do not include chain-of-thought. Do not skip tasks.
For code tasks, put complete code inside the answer string. For JSON-answer tasks, put the requested JSON as a string value.
"""


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str
    allowed_models: list[str]
    request_timeout_seconds: float = 40.0
    max_retries: int = 4
    local_fast_paths: bool = True
    ensemble_mode: str = "hard"  # off | hard | all
    hard_ensemble_size: int = 2
    repair_enabled: bool = True
    batch_solve: bool = True
    batch_candidate_count: int = 1
    batch_max_tokens: int = 9000

    @classmethod
    def from_env(cls) -> "AgentConfig":
        required = ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))
        return cls(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"].rstrip("/"),
            allowed_models=parse_allowed_models(os.environ["ALLOWED_MODELS"]),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "40")),
            max_retries=max(1, int(os.getenv("MAX_RETRIES", "4"))),
            local_fast_paths=os.getenv("ENABLE_LOCAL_FAST_PATHS", "1").strip().lower() in {"1", "true", "yes", "on"},
            ensemble_mode=os.getenv("ENSEMBLE_MODE", "hard").strip().lower(),
            hard_ensemble_size=max(1, int(os.getenv("HARD_ENSEMBLE_SIZE", "2"))),
            repair_enabled=os.getenv("REPAIR_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"},
            batch_solve=os.getenv("BATCH_SOLVE", "1").strip().lower() in {"1", "true", "yes", "on"},
            batch_candidate_count=max(1, int(os.getenv("BATCH_CANDIDATES", "1"))),
            batch_max_tokens=max(1000, int(os.getenv("BATCH_MAX_TOKENS", "9000"))),
        )


class FireworksTrack1Agent:
    """v12 correctness-first agent.

    v12 adds a validated batch solver because Track 1 evaluates a task array.
    It keeps per-task fallback for any missing/invalid answer. This is designed to
    reduce routing/correction mistakes while still satisfying the exact Docker contract.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def answer_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
        if self.config.batch_solve and len(tasks) > 1:
            batch = self._try_batch_solve(tasks)
            if batch is not None:
                return batch
        return [self.answer_task(task) for task in tasks]

    def answer_task(self, task: dict[str, Any]) -> dict[str, str]:
        task_id = str(task.get("task_id", "")) or "missing_task_id"
        prompt = str(task.get("prompt", "")).strip()
        if not prompt:
            return {"task_id": task_id, "answer": ""}

        task_type = classify_task(prompt)

        if self.config.local_fast_paths:
            local = self._try_safe_local(prompt, task_type)
            if local:
                return {"task_id": task_id, "answer": local}

        try:
            answer = self._solve(prompt, task_type)
        except Exception:
            answer = self._fallback(prompt, task_type)

        answer = _final_cleanup(answer, task_type, prompt)

        if self.config.repair_enabled and _needs_repair(answer, task_type, prompt):
            repaired = self._repair(prompt, task_type, answer)
            if repaired:
                answer = _final_cleanup(repaired, task_type, prompt)

        return {"task_id": task_id, "answer": answer.strip() or "I don't know."}

    def _try_batch_solve(self, tasks: list[dict[str, Any]]) -> list[dict[str, str]] | None:
        safe_tasks = []
        for idx, t in enumerate(tasks):
            tid = str(t.get("task_id", f"task_{idx}"))
            prompt = str(t.get("prompt", "")).strip()
            if prompt:
                safe_tasks.append({"task_id": tid, "prompt": prompt})
        if not safe_tasks:
            return []

        # For very large/code-heavy batches, direct per-task solving is safer.
        total_chars = sum(len(t["prompt"]) for t in safe_tasks)
        if total_chars > 16000 or len(safe_tasks) > 30:
            return None

        models = primary_models(self.config.allowed_models, None, limit=self.config.batch_candidate_count)
        for model in models:
            try:
                messages = [
                    {"role": "system", "content": BATCH_SYSTEM},
                    {"role": "user", "content": json.dumps(safe_tasks, ensure_ascii=False)},
                ]
                raw = self._chat_completion(model, messages, self.config.batch_max_tokens)
                parsed = _parse_batch_json(raw)
                if not parsed:
                    continue
                by_id: dict[str, str] = {}
                for item in parsed:
                    if isinstance(item, dict) and "task_id" in item and "answer" in item:
                        by_id[str(item["task_id"])] = str(item["answer"])
                if len(by_id) < max(1, int(len(safe_tasks) * 0.75)):
                    continue

                results: list[dict[str, str]] = []
                missing_or_bad = 0
                for task in safe_tasks:
                    tid = task["task_id"]
                    prompt = task["prompt"]
                    task_type = classify_task(prompt)
                    ans = by_id.get(tid, "")
                    ans = _final_cleanup(ans, task_type, prompt)
                    if not ans or _looks_error(ans) or _needs_repair(ans, task_type, prompt):
                        missing_or_bad += 1
                        results.append(self.answer_task(task))
                    else:
                        results.append({"task_id": tid, "answer": ans})
                # If batch was mostly bad, do not trust it.
                if missing_or_bad > len(safe_tasks) * 0.4:
                    return None
                return results
            except Exception:
                continue
        return None

    def _try_safe_local(self, prompt: str, task_type: TaskType) -> str | None:
        low = prompt.lower()
        # Never override strict JSON/table/exact-length tasks with heuristics except labels/numbers.
        if any(k in low for k in ["json", "schema", "yaml", "xml", "table", "exactly", "at least", "at most", "no more than", "fewer than"]):
            if task_type not in {TaskType.SENTIMENT, TaskType.MATH}:
                return None
        try:
            local = try_solve_locally(prompt, task_type)
        except Exception:
            return None
        if not local:
            return None
        local = _final_cleanup(local, task_type, prompt)
        if task_type == TaskType.SENTIMENT and re.search(r"\b(positive|negative|neutral|mixed|pos|neg)\b", local, flags=re.I):
            return local
        if task_type == TaskType.MATH and re.fullmatch(r"[-+]?(?:\$)?\d[\d,]*(?:\.\d+)?%?(?:\s*[A-Za-z]+)?", local.strip()):
            return local.strip()
        if task_type == TaskType.CODE_GEN and re.search(r"\bdef\s+\w+\s*\(", local) and _requested_name_ok(prompt, local):
            return local
        if task_type == TaskType.CODE_DEBUG and ("len(nums)" in local or ("bug" in local.lower() and "def " in local)):
            return local
        if task_type == TaskType.LOGIC and "=" in local and len(local) < 250:
            return local
        return None

    def _solve(self, prompt: str, task_type: TaskType) -> str:
        hard = task_type in {TaskType.MATH, TaskType.LOGIC, TaskType.CODE_DEBUG, TaskType.CODE_GEN}
        do_ensemble = self.config.ensemble_mode == "all" or (self.config.ensemble_mode == "hard" and hard)
        if do_ensemble:
            candidates = self._candidate_answers(prompt, task_type)
            usable = [_final_cleanup(c, task_type, prompt) for c in candidates if c and not _looks_error(c)]
            usable = _dedupe(usable)
            if len(usable) >= 2:
                return self._select(prompt, task_type, usable)
            if len(usable) == 1:
                return usable[0]
        return self._direct_best(prompt, task_type)

    def _candidate_answers(self, prompt: str, task_type: TaskType) -> list[str]:
        models = primary_models(self.config.allowed_models, task_type, limit=self.config.hard_ensemble_size)
        candidates: list[str] = []
        for model in models:
            try:
                ans = self._direct(prompt, task_type, model)
                if ans and not _looks_error(ans):
                    candidates.append(ans)
            except Exception:
                continue
        return candidates

    def _direct_best(self, prompt: str, task_type: TaskType) -> str:
        for model in ranked_models(self.config.allowed_models, task_type)[: self.config.max_retries]:
            try:
                ans = self._direct(prompt, task_type, model)
                if ans and not _looks_error(ans):
                    return ans
            except Exception:
                continue
        return self._fallback(prompt, task_type)

    def _fallback(self, prompt: str, task_type: TaskType) -> str:
        last = ""
        for model in usable_models(self.config.allowed_models)[: max(1, self.config.max_retries)]:
            try:
                ans = self._direct(prompt, task_type, model, max_tokens=min(_max_tokens(task_type), 900))
                if ans and not _looks_error(ans):
                    return ans
                last = ans or last
            except Exception:
                continue
        return last or "I don't know."

    def _direct(self, prompt: str, task_type: TaskType, model: str, max_tokens: int | None = None) -> str:
        messages = [
            {"role": "system", "content": system_for(task_type)},
            {"role": "user", "content": prompt},
        ]
        return self._chat_completion(model, messages, max_tokens or _max_tokens(task_type))

    def _select(self, prompt: str, task_type: TaskType, candidates: list[str]) -> str:
        model = selector_model(self.config.allowed_models, task_type)
        if not model:
            return candidates[0]
        candidate_text = "\n\n".join(f"Candidate {i+1}:\n{c}" for i, c in enumerate(candidates))
        messages = [
            {"role": "system", "content": SELECTOR_SYSTEM + "\n\n" + system_for(task_type)},
            {"role": "user", "content": f"Original task:\n{prompt}\n\n{candidate_text}\n\nReturn the single best final answer only."},
        ]
        try:
            selected = self._chat_completion(model, messages, _select_tokens(task_type))
            return selected or candidates[0]
        except Exception:
            return candidates[0]

    def _repair(self, prompt: str, task_type: TaskType, bad_answer: str) -> str | None:
        model = selector_model(self.config.allowed_models, task_type)
        if not model:
            return None
        messages = [
            {"role": "system", "content": REPAIR_SYSTEM + "\n\n" + system_for(task_type)},
            {"role": "user", "content": f"Original task:\n{prompt}\n\nInvalid or incomplete answer:\n{bad_answer}\n\nReturn the corrected final answer only."},
        ]
        try:
            return self._chat_completion(model, messages, _select_tokens(task_type))
        except Exception:
            return None

    def _chat_completion(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        url = _chat_completions_url(self.config.base_url)
        payload = {
            "model": model,
            "messages": _prepare_messages(model, messages),
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error = None
        for attempt in range(max(1, self.config.max_retries)):
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.config.request_timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                return _extract_chat_content(parsed)
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")[:700]
                last_error = RuntimeError(f"HTTP {exc.code} from Fireworks proxy: {err_body}")
                if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                    break
            except Exception as exc:
                last_error = exc
            time.sleep(min(4.0, 0.6 * (2 ** attempt)))
        raise RuntimeError(str(last_error) if last_error else "Fireworks request failed")


def _extract_chat_content(parsed: dict[str, Any]) -> str:
    choices = parsed.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in response: {str(parsed)[:700]}")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(f"Malformed choice: {str(parsed)[:700]}")
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return _remove_thinking(content).strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text") or item.get("content")
                    if isinstance(txt, str):
                        parts.append(txt)
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return _remove_thinking("\n".join(parts)).strip()
    if isinstance(choice.get("text"), str):
        return _remove_thinking(choice["text"]).strip()
    raise RuntimeError(f"Cannot extract message content: {str(parsed)[:700]}")


def _parse_batch_json(text: str) -> list[dict[str, Any]] | None:
    if not text:
        return None
    text = _remove_thinking(text).strip()
    # Strip markdown fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text.strip())
    candidates = [text]
    s, e = text.find("["), text.rfind("]")
    if s != -1 and e != -1 and e > s:
        candidates.append(text[s:e+1])
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                return parsed["results"]
        except Exception:
            continue
    return None

# ------------------------- utilities -------------------------

def _prepare_messages(model: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out = [dict(m) for m in messages]
    if is_thinking_model(model) and out and out[-1].get("role") == "user":
        # Gentle, model-agnostic final-only instruction. Do not use nonstandard commands.
        out[-1]["content"] = str(out[-1].get("content", "")) + "\n\nThink privately. Return only the final answer."
    return out


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _max_tokens(task_type: TaskType) -> int:
    return {
        TaskType.FACTUAL: 700,
        TaskType.MATH: 1400,
        TaskType.SENTIMENT: 260,
        TaskType.SUMMARY: 700,
        TaskType.NER: 850,
        TaskType.CODE_DEBUG: 2400,
        TaskType.LOGIC: 2200,
        TaskType.CODE_GEN: 3000,
    }[task_type]


def _select_tokens(task_type: TaskType) -> int:
    return {
        TaskType.FACTUAL: 650,
        TaskType.MATH: 1200,
        TaskType.SENTIMENT: 220,
        TaskType.SUMMARY: 650,
        TaskType.NER: 750,
        TaskType.CODE_DEBUG: 2200,
        TaskType.LOGIC: 1800,
        TaskType.CODE_GEN: 2700,
    }[task_type]


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = re.sub(r"\s+", " ", item.strip().lower())
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _looks_error(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return True
    bad = [
        "unable to produce", "missing required", "traceback", "error from fireworks",
        "no choices in response", "cannot extract", "malformed choice",
    ]
    return any(b in low for b in bad)


def _needs_repair(answer: str, task_type: TaskType, prompt: str) -> bool:
    text = answer.strip()
    lowp = prompt.lower()
    if not text or _looks_error(text):
        return True
    if "json" in lowp and not _extract_json(text):
        return True
    if task_type == TaskType.SENTIMENT:
        labels = _requested_labels(prompt) or ["positive", "negative", "neutral", "mixed"]
        return not any(re.search(rf"\b{re.escape(label)}\b", text, flags=re.I) for label in labels)
    if task_type == TaskType.MATH:
        return not bool(re.search(r"[-+]?\d", text))
    if task_type == TaskType.CODE_GEN:
        return ("function" in lowp or "python" in lowp or "javascript" in lowp) and not _requested_name_ok(prompt, text)
    if task_type == TaskType.CODE_DEBUG:
        return "def " in prompt and "def " not in text and "bug" not in text.lower() and "fix" not in text.lower()
    if task_type == TaskType.NER:
        return len(text) < 5
    return False


def _final_cleanup(answer: str, task_type: TaskType, prompt: str) -> str:
    text = _remove_thinking(answer).strip()
    text = _strip_chat_prefix(text)

    if "json" in prompt.lower():
        extracted = _extract_json(text)
        if extracted:
            return extracted

    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}:
        text = _strip_single_code_fence(text)

    if task_type == TaskType.SENTIMENT:
        text = _cleanup_sentiment(text, prompt)

    lower = prompt.lower()
    if re.search(r"\b(?:choose|select|option)\s+(?:a|b|c|d|e)\b", lower) or "multiple choice" in lower:
        opt = _extract_option(text)
        if opt:
            return opt
    if re.search(r"\b(?:answer yes or no|yes/no|true or false|true/false)\b", lower):
        yn = _extract_booleanish(text)
        if yn:
            return yn
    if _asks_answer_only(lower):
        return _compact_answer_only(text, prompt)
    return text.strip()


def _remove_thinking(text: str) -> str:
    if not text:
        return ""
    out = text.strip()
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.I | re.S).strip()
    if "</think>" in out.lower():
        out = re.split(r"</think>", out, flags=re.I)[-1].strip()
    # Only strip explicit final-answer labels when they appear at the very start/end.
    m = re.fullmatch(r"(?:final\s+answer|answer|result)\s*:\s*(.+)", out, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    return out


def _strip_chat_prefix(text: str) -> str:
    out = text.strip()
    prefixes = [
        r"^sure[,!]?[\s\n]*", r"^of course[,!]?[\s\n]*",
        r"^here(?:'s| is)\s+(?:the\s+)?(?:final\s+)?(?:answer|code|solution)[:\s]*",
        r"^the\s+(?:final\s+)?answer\s+is[:\s]*",
    ]
    for p in prefixes:
        out = re.sub(p, "", out, flags=re.I).strip()
    return out


def _strip_single_code_fence(text: str) -> str:
    m = re.fullmatch(r"\s*```(?:[a-zA-Z0-9_+\-.#]*)?\s*\n(?P<body>.*?)\n```\s*", text, flags=re.S)
    return m.group("body").strip() if m else text.strip()


def _extract_json(text: str) -> str | None:
    candidates: list[str] = []
    for start, end in [("{", "}"), ("[", "]")]:
        s = text.find(start)
        e = text.rfind(end)
        if s != -1 and e != -1 and e > s:
            candidates.append(text[s:e+1])
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            continue
    return None


def _cleanup_sentiment(text: str, prompt: str) -> str:
    labels = _requested_labels(prompt) or ["positive", "negative", "neutral", "mixed"]
    label_map = {label.lower(): label for label in labels}
    # Prefer the first explicit label in the answer, not the first label listed in the prompt.
    label_pat = "|".join(re.escape(x) for x in sorted(labels, key=len, reverse=True))
    patterns = [
        rf"^\s*(?:sentiment|label|classification)?\s*(?:is|=|:)?\s*({label_pat})\b",
        rf"\b(?:sentiment|label|classification)\s*(?:is|=|:)\s*({label_pat})\b",
        rf"\bthe\s+(?:sentiment|label|classification)\s+is\s+({label_pat})\b",
        rf"\b({label_pat})\b",
    ]
    chosen = None
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            chosen = m.group(1)
            break
    if chosen:
        original = label_map.get(chosen.lower(), chosen)
        canonical = original if len(original) <= 3 else original.capitalize()
        if any(k in prompt.lower() for k in ["justify", "explain", "why", "briefly"]):
            rest = re.sub(rf"^\s*(?:sentiment|label|classification)?\s*(?:is|=|:)?\s*{re.escape(chosen)}\s*[:\-—,]*\s*", "", text, flags=re.I | re.S).strip()
            if rest and len(rest) < 180 and rest.lower() != canonical.lower():
                return canonical + " — " + rest
        return canonical
    return text.strip()


def _requested_labels(prompt: str) -> list[str]:
    low = prompt.lower()
    labels: list[str] = []
    for known in ["very positive", "very negative", "positive", "negative", "neutral", "mixed", "pos", "neg", "happy", "sad", "angry"]:
        if re.search(rf"\b{re.escape(known)}\b", low):
            labels.append(known)
    return list(dict.fromkeys(labels))


def _asks_answer_only(lower_prompt: str) -> bool:
    return any(k in lower_prompt for k in [
        "final number only", "number only", "answer only", "just the answer", "only the answer",
        "return only", "output only", "respond only", "single word", "one word",
    ])


def _compact_answer_only(text: str, prompt: str) -> str:
    if classify_task(prompt) == TaskType.MATH:
        nums = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?", text)
        if nums:
            return nums[-1]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1] if len(lines[-1]) <= 120 else lines[0]
    return text


def _extract_option(text: str) -> str | None:
    m = re.search(r"\b([A-E])\b", text.strip(), flags=re.I)
    return m.group(1).upper() if m else None


def _extract_booleanish(text: str) -> str | None:
    m = re.search(r"\b(yes|no|true|false)\b", text.strip(), flags=re.I)
    if not m:
        return None
    val = m.group(1).lower()
    return "True" if val == "true" else "False" if val == "false" else val.capitalize()


def _requested_name_ok(prompt: str, answer: str) -> bool:
    name = _requested_func_or_class(prompt)
    if not name:
        return True
    return re.search(rf"\b(?:def|class|function)\s+{re.escape(name)}\b|\b{re.escape(name)}\s*=", answer) is not None or name in answer


def _requested_func_or_class(prompt: str) -> str | None:
    m = re.search(r"(?:function|method|class)\s+(?:called|named)\s+([A-Za-z_]\w*)", prompt, flags=re.I)
    if m:
        return m.group(1)
    m = re.search(r"(?:def|function|class)\s+([A-Za-z_]\w*)\s*\(", prompt, flags=re.I)
    return m.group(1) if m else None
