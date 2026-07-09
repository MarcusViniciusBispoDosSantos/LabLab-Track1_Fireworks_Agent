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
from .models import is_thinking_model, parse_allowed_models, ranked_models, usable_models
from .prompts import system_for
from .solvers import try_solve_locally


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str
    allowed_models: list[str]
    request_timeout_seconds: float = 25.0
    max_retries: int = 6
    local_fast_paths: bool = False
    self_check_mode: str = "all"
    calibration_enabled: bool = True
    calibration_model_limit: int = 4

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
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "25")),
            max_retries=max(1, int(os.getenv("MAX_RETRIES", "6"))),
            local_fast_paths=os.getenv("ENABLE_LOCAL_FAST_PATHS", "0").strip().lower() in {"1", "true", "yes", "on"},
            self_check_mode=os.getenv("SELF_CHECK_MODE", "all").strip().lower(),
            calibration_enabled=os.getenv("ENABLE_CATEGORY_CALIBRATION", "1").strip().lower() in {"1", "true", "yes", "on"},
            calibration_model_limit=max(1, int(os.getenv("CALIBRATION_MODEL_LIMIT", "4"))),
        )


class FireworksTrack1Agent:
    """Maximum-correctness Track 1 agent.

    v10 changes the strategy from name-based model guesses to category-specific
    model calibration. The hidden benchmark may expose multiple Fireworks models;
    different models often excel at different capabilities. v10 runs a compact
    public calibration set, chooses a model per category, then solves each hidden
    task with the original prompt and a self-check pass.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._category_models: dict[TaskType, list[str]] = {}
        if self.config.calibration_enabled:
            self._calibrate_category_models()

    def answer_task(self, task: dict[str, Any]) -> dict[str, str]:
        task_id = str(task.get("task_id", "")) or "missing_task_id"
        prompt = str(task.get("prompt", "")).strip()
        if not prompt:
            return {"task_id": task_id, "answer": ""}

        task_type = classify_task(prompt)

        if self.config.local_fast_paths:
            try:
                local = try_solve_locally(prompt, task_type)
                if local and _safe_to_use_local(prompt, task_type, local):
                    return {"task_id": task_id, "answer": _final_cleanup(local, task_type, prompt)}
            except Exception:
                pass

        answer = self._solve_with_best_category_model(prompt, task_type)
        answer = _final_cleanup(answer, task_type, prompt)

        if self._should_self_check(prompt, task_type, answer):
            checked = self._self_check(prompt, task_type, answer)
            if checked and not _looks_unusable(checked):
                cleaned = _final_cleanup(checked, task_type, prompt)
                if cleaned:
                    answer = cleaned

        # Last repair if the answer still obviously violates format.
        if _needs_repair(answer, task_type, prompt):
            repaired = self._repair(prompt, task_type, answer)
            if repaired and not _looks_unusable(repaired):
                answer = _final_cleanup(repaired, task_type, prompt)

        return {"task_id": task_id, "answer": answer.strip()}

    # ------------------------- model calibration -------------------------

    def _calibrate_category_models(self) -> None:
        candidates = usable_models(self.config.allowed_models)
        if not candidates:
            return
        # Combine harness order with name heuristic. Limit calls to stay within runtime.
        prelim = ranked_models(candidates, TaskType.FACTUAL)
        ordered: list[str] = []
        for m in prelim + candidates:
            if m not in ordered:
                ordered.append(m)
        candidates = ordered[: min(len(ordered), self.config.calibration_model_limit)]
        if len(candidates) <= 1:
            return

        scores_by_cat: dict[TaskType, list[tuple[int, str]]] = {cat: [] for cat in TaskType}
        for model in candidates:
            try:
                text = self._chat_completion(
                    model,
                    [
                        {"role": "system", "content": "You are a precise benchmark solver. Return compact answers only."},
                        {"role": "user", "content": _CALIBRATION_PROMPT},
                    ],
                    1200,
                )
                scores = _score_category_calibration(text)
            except Exception:
                scores = {cat: -1 for cat in TaskType}
            for cat in TaskType:
                scores_by_cat[cat].append((scores.get(cat, -1), model))

        for cat, scored in scores_by_cat.items():
            scored.sort(key=lambda x: x[0], reverse=True)
            # Keep category-calibrated order but append heuristic fallbacks.
            selected = [m for score, m in scored if score >= 0]
            for m in ranked_models(self.config.allowed_models, cat):
                if m not in selected:
                    selected.append(m)
            self._category_models[cat] = selected

    def _models_for(self, task_type: TaskType) -> list[str]:
        models = self._category_models.get(task_type) or ranked_models(self.config.allowed_models, task_type)
        forced = os.getenv("FORCE_MODEL", "").strip()
        if forced and forced in models:
            return [forced] + [m for m in models if m != forced]
        return models

    # ------------------------- solving -------------------------

    def _solve_with_best_category_model(self, prompt: str, task_type: TaskType) -> str:
        models = self._models_for(task_type)
        last = ""
        for idx, model in enumerate(models[: self.config.max_retries]):
            try:
                ans = self._direct(prompt, task_type, model)
                if ans and not _looks_unusable(ans):
                    # Accept immediately if format is plausible. Otherwise try the next model.
                    if not _needs_repair(_final_cleanup(ans, task_type, prompt), task_type, prompt):
                        return ans
                    last = ans
                else:
                    last = ans or last
            except Exception:
                time.sleep(0.25 * (idx + 1))
        return last.strip() or "Unable to produce a reliable answer."

    def _direct(self, prompt: str, task_type: TaskType, model: str) -> str:
        return self._chat_completion(
            model,
            [
                {"role": "system", "content": system_for(task_type)},
                {"role": "user", "content": prompt},
            ],
            _max_tokens(task_type),
        )

    def _should_self_check(self, prompt: str, task_type: TaskType, answer: str) -> bool:
        mode = self.config.self_check_mode
        if mode in {"0", "false", "off", "none"}:
            return False
        if mode == "hard":
            return task_type in {TaskType.MATH, TaskType.LOGIC, TaskType.CODE_DEBUG, TaskType.CODE_GEN, TaskType.NER}
        # For summaries and short factual tasks, self-check can over-edit strict length.
        if task_type == TaskType.SUMMARY and any(k in prompt.lower() for k in ["one sentence", "exactly", "words"]):
            return False
        return True

    def _self_check(self, prompt: str, task_type: TaskType, draft: str) -> str:
        models = self._models_for(task_type)
        if not models:
            return draft
        # Prefer second category model for independent check when available; otherwise same model.
        model = models[1] if len(models) > 1 else models[0]
        check_prompt = (
            "Verify the draft answer against the original task.\n"
            "If the draft is fully correct and matches the requested format, return it unchanged.\n"
            "If it is wrong, incomplete, too verbose, or format-invalid, return the corrected final answer only.\n\n"
            f"Original task:\n{prompt}\n\nDraft answer:\n{draft}"
        )
        return self._chat_completion(
            model,
            [
                {"role": "system", "content": system_for(task_type)},
                {"role": "user", "content": check_prompt},
            ],
            _repair_tokens(task_type),
        )

    def _repair(self, prompt: str, task_type: TaskType, bad_answer: str) -> str:
        models = self._models_for(task_type)
        if not models:
            return bad_answer
        repair_prompt = (
            "The previous answer does not satisfy the task or requested format.\n"
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
                    return _remove_thinking("".join(parts)).strip()
        if isinstance(choice.get("text"), str):
            return _remove_thinking(choice["text"]).strip()
        raise RuntimeError(f"Cannot extract message content: {raw[:700]}")


# ------------------------- calibration -------------------------

_CALIBRATION_PROMPT = """Answer the following calibration tasks. Use the exact labels C1-C8.
C1 Math: A $120 item is discounted 25% and then 8% tax is added. Final price?
C2 Sentiment: Label as Positive, Negative, Neutral, or Mixed: The interface is clean and fast, but export fails every time.
C3 Summary: Summarize in one sentence: Cloud computing lets companies rent computing resources over the internet instead of buying physical servers. It improves scalability and reduces upfront cost.
C4 NER: Extract entities and labels: On July 4, 2026, Ana Silva joined AMD in Austin, Texas.
C5 Debug: Fix this Python bug: def avg(nums): return sum(nums) / len(num)
C6 Logic: Alice, Bob, and Cara each own one pet: cat, dog, or bird. Alice does not own the dog. Bob does not own the cat. Cara owns the bird. Who owns each pet?
C7 Code: Write Python function is_even(n) returning True if n is even.
C8 Factual: In one sentence, what is HTTP?
"""


def _score_category_calibration(text: str) -> dict[TaskType, int]:
    low = text.lower()
    scores = {cat: 0 for cat in TaskType}
    if re.search(r"\b97\.?(?:20)?\b", low) or "$97.20" in text:
        scores[TaskType.MATH] += 3
    if "mixed" in low:
        scores[TaskType.SENTIMENT] += 3
    if "rent" in low and ("computing" in low or "resources" in low) and ("scal" in low or "cost" in low):
        scores[TaskType.SUMMARY] += 3
    if all(x in low for x in ["ana", "amd", "austin"]) and ("date" in low or "july 4" in low):
        scores[TaskType.NER] += 3
    if "len(nums)" in text or ("len(num)" in text and "bug" in low and "nums" in low):
        scores[TaskType.CODE_DEBUG] += 3
    if all(x in low for x in ["alice", "cat", "bob", "dog", "cara", "bird"]):
        scores[TaskType.LOGIC] += 3
    if "def is_even" in text and ("% 2" in text or "& 1" in text or "mod" in low):
        scores[TaskType.CODE_GEN] += 3
    if "http" in low and ("protocol" in low or "web" in low or "browser" in low or "server" in low):
        scores[TaskType.FACTUAL] += 3
    # General formatting bonus: answer all labels rather than rambling.
    for i in range(1, 9):
        if f"c{i}" in low:
            for cat in TaskType:
                scores[cat] += 1
            break
    return scores


# ------------------------- utilities -------------------------

def _prepare_messages(model: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out = [dict(m) for m in messages]
    if is_thinking_model(model) and out and out[-1].get("role") == "user":
        out[-1]["content"] = str(out[-1].get("content", "")) + "\n\nThink privately. Do not include hidden reasoning. Return only the final answer."
    return out


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _max_tokens(task_type: TaskType) -> int:
    return {
        TaskType.FACTUAL: 700,
        TaskType.MATH: 1200,
        TaskType.SENTIMENT: 260,
        TaskType.SUMMARY: 700,
        TaskType.NER: 800,
        TaskType.CODE_DEBUG: 2600,
        TaskType.LOGIC: 1800,
        TaskType.CODE_GEN: 3000,
    }[task_type]


def _repair_tokens(task_type: TaskType) -> int:
    return {
        TaskType.FACTUAL: 600,
        TaskType.MATH: 1000,
        TaskType.SENTIMENT: 220,
        TaskType.SUMMARY: 600,
        TaskType.NER: 700,
        TaskType.CODE_DEBUG: 2200,
        TaskType.LOGIC: 1500,
        TaskType.CODE_GEN: 2600,
    }[task_type]


def _safe_to_use_local(prompt: str, task_type: TaskType, answer: str) -> bool:
    low = prompt.lower()
    if any(k in low for k in ["json", "schema", "strict", "explain", "show your", "prove", "why", "table", "format"]):
        return False
    if task_type == TaskType.SENTIMENT:
        return answer.split()[0].strip("—:-").lower() in {"positive", "negative", "neutral", "mixed"} and len(answer) < 220
    if task_type == TaskType.MATH:
        return bool(re.fullmatch(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?(?:\s*[A-Za-z]+)?", answer.strip()))
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
    if task_type == TaskType.MATH:
        if not re.search(r"[-+]?\d", text):
            return True
    if task_type == TaskType.CODE_GEN:
        name = _requested_func_or_class(prompt)
        if name and name not in text:
            return True
        if any(k in lowp for k in ["python function", "write a function", "implement a function"]) and not re.search(r"\bdef\s+\w+\s*\(", text):
            return True
    if task_type == TaskType.CODE_DEBUG:
        if "def " in prompt and "def " not in text and "bug" not in text.lower() and "fix" not in text.lower():
            return True
    if task_type == TaskType.NER and len(text) < 6:
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
    if re.search(r"\b(?:choose|select|option)\s+(?:a|b|c|d|e)\b", lower) or "multiple choice" in lower:
        opt = _extract_option(text)
        if opt:
            return opt
    if re.search(r"\b(?:answer yes or no|yes/no|true or false|true/false)\b", lower):
        yn = _extract_booleanish(text)
        if yn:
            return yn
    if _asks_answer_only(lower):
        text = _compact_answer_only(text, prompt)
    return text.strip()


def _remove_thinking(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1].strip()
    m = re.search(r"(?:^|\n)\s*(?:final answer|answer|result)\s*:\s*(.+)$", text, flags=re.I | re.S)
    if m:
        cand = m.group(1).strip()
        if cand:
            return cand
    return text


def _strip_chat_prefix(text: str) -> str:
    out = text.strip()
    for p in [
        r"^sure[,!]?\s*", r"^of course[,!]?\s*",
        r"^here(?:'s| is)\s+(?:the\s+)?(?:final\s+)?(?:answer|code|solution)[:\s]*",
        r"^the\s+(?:final\s+)?answer\s+is[:\s]*",
    ]:
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
    if m:
        return m.group(1)
    m = re.search(r"(?:def|function|class)\s+([A-Za-z_]\w*)\s*\(", prompt, flags=re.I)
    return m.group(1) if m else None
