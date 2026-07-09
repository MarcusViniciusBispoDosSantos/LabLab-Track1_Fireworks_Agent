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
from .models import best_selector_model, diverse_models, is_thinking_model, parse_allowed_models, ranked_models
from .prompts import CANDIDATE_SYSTEM, SELECTION_SYSTEM, TASK_SYSTEMS
from .solvers import try_solve_locally


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str
    allowed_models: list[str]
    request_timeout_seconds: float = 28.0
    max_retries: int = 3
    strategy: str = "ensemble_hard"  # direct | verify_hard | ensemble_hard | ensemble_all
    local_fast_paths: bool = True

    @classmethod
    def from_env(cls) -> "AgentConfig":
        required = ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))

        strategy = os.getenv("ACCURACY_STRATEGY", "ensemble_hard").strip().lower()
        # Backward compatibility with earlier vars.
        verify_mode = os.getenv("VERIFY_MODE", "").strip().lower()
        if verify_mode == "all" and "ACCURACY_STRATEGY" not in os.environ:
            strategy = "ensemble_all"
        elif verify_mode == "hard" and "ACCURACY_STRATEGY" not in os.environ:
            strategy = "ensemble_hard"
        elif verify_mode == "none" and "ACCURACY_STRATEGY" not in os.environ:
            strategy = "direct"
        if strategy not in {"direct", "verify_hard", "ensemble_hard", "ensemble_all"}:
            strategy = "ensemble_hard"

        return cls(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"].rstrip("/"),
            allowed_models=parse_allowed_models(os.environ["ALLOWED_MODELS"]),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "28")),
            max_retries=max(0, int(os.getenv("MAX_RETRIES", "3"))),
            strategy=strategy,
            local_fast_paths=os.getenv("ENABLE_LOCAL_FAST_PATHS", "1").strip().lower() in {"1", "true", "yes", "on"},
        )


class FireworksTrack1Agent:
    def __init__(self, config: AgentConfig):
        self.config = config

    def answer_task(self, task: dict[str, Any]) -> dict[str, str]:
        task_id = str(task.get("task_id", "")) or "missing_task_id"
        prompt = str(task.get("prompt", "")).strip()
        if not prompt:
            return {"task_id": task_id, "answer": ""}

        task_type = classify_task(prompt)

        # Use deterministic solvers only when they return a high-confidence result.
        # This recovers easy benchmark points without spending model tokens. The
        # solvers are conservative and fall through for varied/ambiguous prompts.
        if self.config.local_fast_paths:
            try:
                local = try_solve_locally(prompt, task_type)
                if local and _looks_high_confidence_local(prompt, task_type, local):
                    return {"task_id": task_id, "answer": _clean_answer(local, task_type, prompt)}
            except Exception:
                pass

        try:
            answer = self._solve_with_strategy(prompt, task_type)
        except Exception:
            # Final fallback: one direct call to best available factual/general model.
            try:
                answer = self._direct_solve(prompt, TaskType.FACTUAL)
            except Exception:
                answer = "Unable to produce a reliable answer."

        return {"task_id": task_id, "answer": _clean_answer(answer, task_type, prompt) or answer.strip()}

    def _solve_with_strategy(self, prompt: str, task_type: TaskType) -> str:
        hard = task_type in {TaskType.MATH, TaskType.LOGIC, TaskType.CODE_DEBUG, TaskType.CODE_GEN}
        if self.config.strategy == "direct":
            return self._direct_solve(prompt, task_type)
        if self.config.strategy == "verify_hard":
            draft = self._direct_solve(prompt, task_type)
            return self._select_final(prompt, task_type, [draft]) if hard else draft
        if self.config.strategy == "ensemble_all" or (self.config.strategy == "ensemble_hard" and hard):
            return self._ensemble_solve(prompt, task_type)
        return self._direct_solve(prompt, task_type)

    def _direct_solve(self, prompt: str, task_type: TaskType) -> str:
        messages = [
            {"role": "system", "content": TASK_SYSTEMS[task_type]},
            {"role": "user", "content": prompt},
        ]
        return self._call_best_available(messages, task_type, _max_tokens_for(task_type))

    def _ensemble_solve(self, prompt: str, task_type: TaskType) -> str:
        # For hard categories, produce candidates from diverse strong models, then
        # select/synthesize one final answer. This costs more tokens but is designed
        # to clear the accuracy gate first.
        candidate_models = diverse_models(self.config.allowed_models, task_type, limit=_ensemble_size_for(task_type))
        if not candidate_models:
            return self._direct_solve(prompt, task_type)

        candidates: list[str] = []
        for model in candidate_models:
            try:
                messages = [
                    {"role": "system", "content": CANDIDATE_SYSTEM + "\n" + TASK_SYSTEMS[task_type]},
                    {"role": "user", "content": prompt},
                ]
                content = self._chat_completion(model, messages, _max_tokens_for(task_type))
                content = _remove_thinking(content).strip()
                if content and content not in candidates:
                    candidates.append(content)
            except Exception:
                continue

        if not candidates:
            return self._direct_solve(prompt, task_type)
        if len(candidates) == 1 and task_type not in {TaskType.MATH, TaskType.LOGIC}:
            return candidates[0]
        return self._select_final(prompt, task_type, candidates)

    def _select_final(self, prompt: str, task_type: TaskType, candidates: list[str]) -> str:
        selector = best_selector_model(self.config.allowed_models, task_type)
        if not selector:
            return candidates[0] if candidates else ""
        candidate_text = "\n\n".join(f"Candidate {i+1}:\n{c}" for i, c in enumerate(candidates))
        messages = [
            {"role": "system", "content": SELECTION_SYSTEM + "\n" + TASK_SYSTEMS[task_type]},
            {"role": "user", "content": f"Original user task:\n{prompt}\n\n{candidate_text}\n\nReturn the single best final answer only."},
        ]
        try:
            return self._chat_completion(selector, messages, _select_max_tokens_for(task_type))
        except Exception:
            return candidates[0] if candidates else ""

    def _call_best_available(self, messages: list[dict[str, str]], task_type: TaskType, max_tokens: int) -> str:
        candidates = ranked_models(self.config.allowed_models, task_type)
        if not candidates:
            raise RuntimeError("No usable models found in ALLOWED_MODELS")
        last_error: Exception | None = None
        max_attempts = max(1, min(len(candidates), self.config.max_retries + 1))
        for i, model in enumerate(candidates[:max_attempts], start=1):
            try:
                content = self._chat_completion(model=model, messages=messages, max_tokens=max_tokens)
                content = _remove_thinking(content).strip()
                if content:
                    return content
                last_error = RuntimeError(f"Empty content from {model}")
            except Exception as exc:
                last_error = exc
                time.sleep(min(0.6 * i, 2.0))
        raise RuntimeError(f"All model attempts failed. Last error: {last_error}")

    def _chat_completion(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        url = _chat_completions_url(self.config.base_url)
        messages_to_send = _prepare_messages_for_model(model, messages)
        payload = {
            "model": model,
            "messages": messages_to_send,
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
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError(f"Malformed choice: {raw[:700]}")

        message = first.get("message")
        if isinstance(message, dict):
            for key in ("content", "text"):
                content = message.get(key)
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            for k in ("text", "content"):
                                if isinstance(item.get(k), str):
                                    parts.append(item[k])
                                    break
                        elif isinstance(item, str):
                            parts.append(item)
                    if parts:
                        return "\n".join(parts)
        text = first.get("text")
        if isinstance(text, str):
            return text
        raise RuntimeError(f"Could not parse assistant content: {raw[:700]}")


def _prepare_messages_for_model(model: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out = [dict(m) for m in messages]
    # Some reasoning models support /no_think; do not force it for math/logic
    # candidates because hidden benchmarks may need reasoning. We still demand a
    # final-only answer in the prompt and strip tagged thinking afterward.
    if is_thinking_model(model) and out and out[-1].get("role") == "user":
        out[-1]["content"] = str(out[-1].get("content", "")) + "\n\nReturn only the final answer after private reasoning."
    return out


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _max_tokens_for(task_type: TaskType) -> int:
    return {
        TaskType.FACTUAL: 800,
        TaskType.MATH: 1800,
        TaskType.SENTIMENT: 260,
        TaskType.SUMMARY: 650,
        TaskType.NER: 850,
        TaskType.CODE_DEBUG: 2600,
        TaskType.LOGIC: 2200,
        TaskType.CODE_GEN: 3000,
    }[task_type]


def _select_max_tokens_for(task_type: TaskType) -> int:
    return {
        TaskType.FACTUAL: 700,
        TaskType.MATH: 1400,
        TaskType.SENTIMENT: 220,
        TaskType.SUMMARY: 550,
        TaskType.NER: 750,
        TaskType.CODE_DEBUG: 2400,
        TaskType.LOGIC: 1600,
        TaskType.CODE_GEN: 2800,
    }[task_type]


def _ensemble_size_for(task_type: TaskType) -> int:
    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG, TaskType.MATH, TaskType.LOGIC}:
        return 3
    return 2


def _looks_high_confidence_local(prompt: str, task_type: TaskType, answer: str) -> bool:
    # Do not use local solver when prompt explicitly asks for a detailed explanation,
    # proof, JSON schema, or complex formatting beyond the solver's simple output.
    low = prompt.lower()
    if any(k in low for k in ["json", "schema", "explain in detail", "prove", "show your work", "step by step"]):
        return False
    if task_type == TaskType.MATH:
        # numeric answer only is safe for many simple arithmetic/percentage tasks.
        return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*[A-Za-z%$]+)?", answer.strip())) or len(answer) < 80
    if task_type == TaskType.SENTIMENT:
        return answer.strip().split()[0].strip("—:-").lower() in {"positive", "negative", "neutral", "mixed"}
    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG, TaskType.LOGIC}:
        return len(answer.strip()) > 0 and len(answer) < 2500
    # Avoid local NER by default; LLM handles varied entity schemas better.
    return False


def _clean_answer(answer: str, task_type: TaskType, prompt: str = "") -> str:
    text = _remove_thinking(answer).strip()
    text = re.sub(r"^\s*(final answer|answer|result)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = _remove_chatty_prefixes(text)

    if task_type == TaskType.CODE_GEN:
        text = _strip_single_code_fence(text)
        text = re.sub(r"^(?:here is|here's)\s+(?:the\s+)?(?:correct\s+)?(?:code|function|implementation)\s*:?\s*", "", text, flags=re.I).strip()
    elif task_type == TaskType.CODE_DEBUG:
        text = _strip_single_code_fence(text)

    if task_type == TaskType.SENTIMENT:
        # Make label-first if the model included a preamble.
        m = re.search(r"\b(positive|negative|neutral|mixed)\b", text, flags=re.I)
        if m and not text.lower().startswith(m.group(1).lower()):
            label = m.group(1).capitalize()
            # Preserve short justification if present.
            if len(text) < 180 and ("because" in text.lower() or "—" in text or "-" in text):
                text = label + " — " + re.sub(r"^.*?\b(?:positive|negative|neutral|mixed)\b\s*[:\-—,]*\s*", "", text, flags=re.I).strip()
            else:
                text = label
    return text.strip()


def _remove_thinking(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()
    # Prefer explicit final-answer sections, but only if they are not empty.
    matches = list(re.finditer(r"(?:final answer|answer)\s*:\s*(.+)$", text, flags=re.I | re.S))
    if matches:
        candidate = matches[-1].group(1).strip()
        if candidate:
            return candidate
    return text


def _remove_chatty_prefixes(text: str) -> str:
    prefixes = [
        r"^sure[,!]?\s*", r"^of course[,!]?\s*", r"^here is the answer[:\s]*",
        r"^the answer is[:\s]*", r"^the corrected answer is[:\s]*",
    ]
    out = text.strip()
    for p in prefixes:
        out = re.sub(p, "", out, flags=re.I).strip()
    return out


def _strip_single_code_fence(text: str) -> str:
    m = re.fullmatch(r"\s*```(?:[a-zA-Z0-9_+\-.#]*)?\s*\n(?P<code>.*?)\n```\s*", text, flags=re.DOTALL)
    if m:
        return m.group("code").strip()
    return text
