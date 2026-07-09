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
from .models import is_thinking_model, parse_allowed_models, ranked_models
from .prompts import system_for
from .solvers import try_solve_locally


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str
    allowed_models: list[str]
    request_timeout_seconds: float = 26.0
    max_retries: int = 5
    local_fast_paths: bool = True
    retry_invalid_outputs: bool = True

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
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "26")),
            max_retries=max(1, int(os.getenv("MAX_RETRIES", "5"))),
            local_fast_paths=os.getenv("ENABLE_LOCAL_FAST_PATHS", "1").strip().lower() in {"1", "true", "yes", "on"},
            retry_invalid_outputs=os.getenv("RETRY_INVALID_OUTPUTS", "1").strip().lower() in {"1", "true", "yes", "on"},
        )


class FireworksTrack1Agent:
    """Accuracy-first but runtime-safe Track 1 agent.

    v9 target-85 strategy:
    - Use deterministic local solvers only for very high-confidence exact cases.
    - Otherwise send the original hidden prompt directly to the strongest allowed model.
    - Do not run a verifier on every task; prior versions often corrupted good answers
      or exceeded runtime. Retry only when output is clearly invalid for the prompt.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._calibrated_model: str | None = None
        self._calibrated_code_model: str | None = None
        if os.getenv("ENABLE_MODEL_CALIBRATION", "1").strip().lower() in {"1", "true", "yes", "on"}:
            self._run_model_calibration()


    def _preferred_model(self, task_type: TaskType) -> str | None:
        if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG} and self._calibrated_code_model:
            return self._calibrated_code_model
        return self._calibrated_model

    def _run_model_calibration(self) -> None:
        """Choose the most accurate model from ALLOWED_MODELS using tiny known tasks.

        This does not cache hidden evaluation answers. It only detects which allowed
        model follows instructions best in the current harness. Previous versions
        likely lost accuracy by choosing a weaker model from ALLOWED_MODELS.
        """
        models = ranked_models(self.config.allowed_models, TaskType.FACTUAL)
        if len(models) <= 1:
            return
        limit = max(1, min(int(os.getenv("CALIBRATION_MODEL_LIMIT", "4")), len(models)))
        candidates = models[:limit]
        best_model = candidates[0]
        best_score = -1
        best_code_model = candidates[0]
        best_code_score = -1
        for model in candidates:
            try:
                text = self._chat_completion(model, [
                    {"role": "system", "content": "You are a precise evaluator. Return concise answers only."},
                    {"role": "user", "content": _CALIBRATION_PROMPT},
                ], 500)
                score, code_score = _score_calibration(text)
            except Exception:
                score, code_score = -1, -1
            if score > best_score:
                best_score = score
                best_model = model
            if code_score > best_code_score:
                best_code_score = code_score
                best_code_model = model
        # Only override if the model demonstrates at least basic correctness.
        if best_score >= 2:
            self._calibrated_model = best_model
        if best_code_score >= 1:
            self._calibrated_code_model = best_code_model

    def answer_task(self, task: dict[str, Any]) -> dict[str, str]:
        task_id = str(task.get("task_id", "")) or "missing_task_id"
        prompt = str(task.get("prompt", "")).strip()
        if not prompt:
            return {"task_id": task_id, "answer": ""}

        task_type = classify_task(prompt)

        # High-confidence deterministic solvers. These avoid weak-model mistakes on
        # common exact tasks, but only if the local answer is provably format-safe.
        if self.config.local_fast_paths:
            try:
                local = try_solve_locally(prompt, task_type)
                if local and _safe_to_use_local(prompt, task_type, local):
                    return {"task_id": task_id, "answer": _final_cleanup(local, task_type, prompt)}
            except Exception:
                pass

        answer = self._solve_direct_with_fallback(prompt, task_type)
        answer = _final_cleanup(answer, task_type, prompt)

        # Targeted repair only when the final format is obviously wrong. This gives
        # accuracy benefits without doubling calls on every hidden task.
        if self.config.retry_invalid_outputs and _needs_repair(answer, task_type, prompt):
            repaired = self._repair(prompt, task_type, answer)
            if repaired and not _looks_unusable(repaired):
                answer = _final_cleanup(repaired, task_type, prompt)

        return {"task_id": task_id, "answer": answer}

    def _solve_direct_with_fallback(self, prompt: str, task_type: TaskType) -> str:
        models = ranked_models(self.config.allowed_models, task_type)
        preferred = self._preferred_model(task_type)
        if preferred and preferred in models:
            models = [preferred] + [m for m in models if m != preferred]
        if not models:
            return "Unable to produce a reliable answer."
        last = ""
        tried = 0
        for model in models:
            if tried >= self.config.max_retries:
                break
            tried += 1
            try:
                ans = self._direct(prompt, task_type, model)
                if ans and not _looks_unusable(ans):
                    return ans
                last = ans or last
            except Exception:
                time.sleep(0.4 * tried)
        return last.strip() or "Unable to produce a reliable answer."

    def _direct(self, prompt: str, task_type: TaskType, model: str) -> str:
        messages = [
            {"role": "system", "content": system_for(task_type)},
            {"role": "user", "content": prompt},
        ]
        return self._chat_completion(model, messages, _max_tokens(task_type))

    def _repair(self, prompt: str, task_type: TaskType, bad_answer: str) -> str:
        models = ranked_models(self.config.allowed_models, task_type)
        if not models:
            return bad_answer
        repair_prompt = (
            "The previous answer did not satisfy the requested format or was incomplete.\n"
            "Return the corrected final answer only.\n\n"
            f"Original task:\n{prompt}\n\nPrevious answer:\n{bad_answer}"
        )
        for model in models[:2]:
            try:
                return self._chat_completion(
                    model,
                    [
                        {"role": "system", "content": system_for(task_type)},
                        {"role": "user", "content": repair_prompt},
                    ],
                    _repair_tokens(task_type),
                )
            except Exception:
                continue
        return bad_answer

    def _chat_completion(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        url = _chat_completions_url(self.config.base_url)
        payload = {
            "model": model,
            "messages": _prepare_messages(model, messages),
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
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
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from Fireworks proxy: {body[:700]}") from exc

        parsed = json.loads(raw)
        choices = parsed.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in response: {raw[:700]}")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError(f"Malformed choice: {raw[:700]}")
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
        text = choice.get("text")
        if isinstance(text, str):
            return _remove_thinking(text).strip()
        raise RuntimeError(f"Could not parse assistant content: {raw[:700]}")



_CALIBRATION_PROMPT = """Solve these calibration tasks. Return plain text with four labeled lines: MATH, SENTIMENT, DEBUG, LOGIC.
MATH: A store has 100 users. It grows by 20% and then loses 10%. What is the final number?
SENTIMENT: Classify as Positive, Negative, Neutral, or Mixed: The interface is fast and clean, but it crashes often.
DEBUG: In Python, what is the bug and fix? def average(nums): return sum(nums) / len(num)
LOGIC: Alice, Bob, and Cara each own one pet: cat, dog, or bird. Alice does not own the dog. Bob does not own the cat. Cara owns the bird. Who owns each pet?
"""


def _score_calibration(text: str) -> tuple[int, int]:
    low = text.lower()
    score = 0
    code_score = 0
    if re.search(r"\b108(?:\.0+)?\b", low):
        score += 1
    if "mixed" in low:
        score += 1
    if "len(nums)" in text or ("num" in low and "undefined" in low):
        score += 1
        code_score += 1
    if all(x in low for x in ["alice", "cat", "bob", "dog", "cara", "bird"]):
        score += 1
    return score, code_score

def _prepare_messages(model: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out = [dict(m) for m in messages]
    if is_thinking_model(model) and out and out[-1].get("role") == "user":
        # Do not use nonstandard /no_think tokens; some model families are harmed by it.
        out[-1]["content"] = str(out[-1].get("content", "")) + "\n\nThink privately. Return only the final answer requested."
    return out


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _max_tokens(task_type: TaskType) -> int:
    # Keep enough room for code, but avoid long rambling that hurts runtime and judge matching.
    return {
        TaskType.FACTUAL: 700,
        TaskType.MATH: 1100,
        TaskType.SENTIMENT: 220,
        TaskType.SUMMARY: 650,
        TaskType.NER: 700,
        TaskType.CODE_DEBUG: 2200,
        TaskType.LOGIC: 1500,
        TaskType.CODE_GEN: 2600,
    }[task_type]


def _repair_tokens(task_type: TaskType) -> int:
    return {
        TaskType.FACTUAL: 550,
        TaskType.MATH: 900,
        TaskType.SENTIMENT: 180,
        TaskType.SUMMARY: 500,
        TaskType.NER: 550,
        TaskType.CODE_DEBUG: 1800,
        TaskType.LOGIC: 1200,
        TaskType.CODE_GEN: 2200,
    }[task_type]


def _safe_to_use_local(prompt: str, task_type: TaskType, answer: str) -> bool:
    low = prompt.lower()
    # Do not use local shortcuts when the benchmark asks for explanations,
    # strict schemas, unusual formatting, or hidden nuance.
    if any(k in low for k in ["json", "schema", "strict", "explain", "show your", "prove", "why", "table", "format"]):
        return False
    if task_type == TaskType.SENTIMENT:
        return answer.split()[0].strip("—:-").lower() in {"positive", "negative", "neutral", "mixed"} and len(answer) < 220
    if task_type == TaskType.CODE_GEN:
        return "def " in answer and len(answer) < 1600
    if task_type == TaskType.CODE_DEBUG:
        return "def " in answer and len(answer) < 1800
    if task_type == TaskType.MATH:
        return bool(re.fullmatch(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?(?:\s*[A-Za-z]+)?", answer.strip()))
    if task_type == TaskType.LOGIC:
        return len(answer) < 350 and ("=" in answer or ":" in answer)
    return False


def _needs_repair(answer: str, task_type: TaskType, prompt: str) -> bool:
    text = answer.strip()
    lowp = prompt.lower()
    if not text or _looks_unusable(text):
        return True
    if "json" in lowp and not _extract_json(text):
        return True
    if task_type == TaskType.SENTIMENT:
        labels = _requested_labels(prompt) or ["positive", "negative", "neutral", "mixed"]
        if not any(re.search(rf"\b{re.escape(l)}\b", text, flags=re.I) for l in labels):
            return True
    if task_type == TaskType.CODE_GEN:
        name = _requested_func_or_class(prompt)
        if name and name not in text:
            return True
        if any(k in lowp for k in ["python", "function", "method"]) and not re.search(r"\bdef\s+\w+\s*\(", text):
            return True
    if task_type == TaskType.CODE_DEBUG:
        if "def " in prompt and "def " not in text and "bug" not in text.lower() and "fix" not in text.lower():
            return True
    if task_type == TaskType.NER and len(text) < 3:
        return True
    return False


def _looks_unusable(text: str) -> bool:
    low = text.lower().strip()
    bad = [
        "unable to produce", "i cannot", "i can't", "as an ai", "i don't have enough",
        "missing required", "http ", "traceback", "error from fireworks", "no choices",
    ]
    return any(b in low for b in bad)


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
    if _asks_answer_only(lower):
        text = _compact_answer_only(text, prompt)
    if re.search(r"\b(?:choose|select|option)\s+(?:a|b|c|d|e)\b", lower) or "multiple choice" in lower:
        opt = _extract_option(text)
        if opt:
            return opt
    if re.search(r"\b(?:answer yes or no|yes/no|true or false|true/false)\b", lower):
        yn = _extract_booleanish(text)
        if yn:
            return yn
    return text.strip()


def _remove_thinking(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1].strip()
    # Only strip explicit final-answer markers at the beginning of a line. Older
    # versions used a broader regex that could accidentally delete useful content.
    m = re.search(r"(?:^|\n)\s*(?:final answer|answer|result)\s*:\s*(.+)$", text, flags=re.I | re.S)
    if m:
        cand = m.group(1).strip()
        if cand:
            return cand
    return text


def _strip_chat_prefix(text: str) -> str:
    out = text.strip()
    patterns = [
        r"^sure[,!]?\s*", r"^of course[,!]?\s*",
        r"^here(?:'s| is)\s+(?:the\s+)?(?:final\s+)?(?:answer|code|solution)[:\s]*",
        r"^the\s+(?:final\s+)?answer\s+is[:\s]*",
    ]
    for p in patterns:
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
    label_pat = "|".join(re.escape(x) for x in labels)
    chosen: str | None = None
    for pat in [
        rf"^\s*({label_pat})\b",
        rf"\b(?:sentiment|label|classification)\s*(?:is|=|:)\s*({label_pat})\b",
        rf"\bthe\s+(?:sentiment|label|classification)\s+is\s+({label_pat})\b",
    ]:
        m = re.search(pat, text, flags=re.I)
        if m:
            chosen = m.group(1)
            break
    if chosen is None:
        low_text = text.lower()
        if "mixed" in low_text:
            chosen = "mixed"
        elif "positive" in low_text and "negative" in low_text and any(l.lower() == "mixed" for l in labels):
            chosen = "mixed"
        else:
            for label in labels:
                if re.search(rf"\b{re.escape(label)}\b", text, flags=re.I):
                    chosen = label
                    break
    if chosen:
        canonical = next((l for l in labels if l.lower() == chosen.lower()), chosen)
        # Preserve requested casing for binary labels when provided; otherwise title case.
        canonical_out = canonical if canonical != canonical.lower() else canonical.capitalize()
        if any(k in prompt.lower() for k in ["justify", "explain", "why", "briefly"]):
            rest = re.sub(rf"^\s*(?:sentiment|label|classification)?\s*(?:is|=|:)?\s*{re.escape(chosen)}\s*[:\-—,]*\s*", "", text, flags=re.I | re.S).strip()
            if rest and len(rest) < 180 and rest.lower() != canonical_out.lower():
                return canonical_out + " — " + rest
        return canonical_out
    return text


def _requested_labels(prompt: str) -> list[str]:
    low = prompt.lower()
    labels: list[str] = []
    # Capture quoted or enumerated labels in the prompt.
    for known in ["very positive", "very negative", "positive", "negative", "neutral", "mixed", "pos", "neg"]:
        if re.search(rf"\b{re.escape(known)}\b", low):
            labels.append(known)
    return list(dict.fromkeys(labels))


def _asks_answer_only(lower_prompt: str) -> bool:
    return any(k in lower_prompt for k in [
        "final number only", "number only", "answer only", "just the answer", "only the answer",
        "return only", "output only", "respond only",
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
    m = re.search(r"\b([A-E])\b", text.strip())
    return m.group(1) if m else None


def _extract_booleanish(text: str) -> str | None:
    m = re.search(r"\b(yes|no|true|false)\b", text.strip(), flags=re.I)
    return m.group(1).capitalize() if m else None


def _requested_func_or_class(prompt: str) -> str | None:
    m = re.search(r"(?:function|method|class)\s+(?:called|named)\s+([A-Za-z_]\w*)", prompt, flags=re.I)
    return m.group(1) if m else None
