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
from .models import is_thinking_model, parse_allowed_models, ranked_models, top_non_thinking_models
from .prompts import TASK_SYSTEMS, VERIFIER_SYSTEM
from .solvers import try_solve_locally


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str
    allowed_models: list[str]
    request_timeout_seconds: float = 29.0
    max_retries: int = 4
    verify_mode: str = "hard"  # none | hard | all
    local_fast_paths: bool = False
    use_ensemble_for_code: bool = False

    @classmethod
    def from_env(cls) -> "AgentConfig":
        required = ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))

        verify_mode = os.getenv("VERIFY_MODE", "hard").strip().lower()
        legacy = os.getenv("VERIFY_HARD_TASKS", "").strip().lower()
        if legacy in {"1", "true", "yes", "on"} and "VERIFY_MODE" not in os.environ:
            verify_mode = "hard"
        if legacy in {"0", "false", "no", "off"} and "VERIFY_MODE" not in os.environ:
            verify_mode = "none"
        if verify_mode not in {"none", "hard", "all"}:
            verify_mode = "hard"

        return cls(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"].rstrip("/"),
            allowed_models=parse_allowed_models(os.environ["ALLOWED_MODELS"]),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "29")),
            max_retries=max(0, int(os.getenv("MAX_RETRIES", "4"))),
            verify_mode=verify_mode,
            # Critical v6 default: hidden prompts should be solved by the strongest LLM,
            # not by brittle regex shortcuts. Keep deterministic solvers opt-in only.
            local_fast_paths=os.getenv("ENABLE_LOCAL_FAST_PATHS", "0").strip().lower() in {"1", "true", "yes", "on"},
            use_ensemble_for_code=os.getenv("USE_ENSEMBLE_FOR_CODE", "0").strip().lower() in {"1", "true", "yes", "on"},
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

        if self.config.local_fast_paths:
            local = try_solve_locally(prompt, task_type)
            if local:
                return {"task_id": task_id, "answer": _clean_answer(local, task_type, prompt)}

        try:
            if self.config.use_ensemble_for_code and task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}:
                answer = self._generate_code_ensemble_answer(prompt, task_type)
            else:
                answer = self._generate_answer(prompt, task_type)
        except Exception:
            # Last-resort universal factual route. This preserves JSON output, but still uses Fireworks.
            try:
                answer = self._generate_answer(prompt, TaskType.FACTUAL)
            except Exception:
                answer = "Unable to produce a reliable answer."

        if self._should_verify(task_type) and answer != "Unable to produce a reliable answer.":
            try:
                answer = self._verify_answer(prompt, answer, task_type)
            except Exception:
                pass

        return {"task_id": task_id, "answer": _clean_answer(answer, task_type, prompt) or answer.strip()}

    def _should_verify(self, task_type: TaskType) -> bool:
        if self.config.verify_mode == "none":
            return False
        if self.config.verify_mode == "all":
            return True
        return task_type in {TaskType.MATH, TaskType.LOGIC, TaskType.CODE_DEBUG, TaskType.CODE_GEN}

    def _generate_answer(self, prompt: str, task_type: TaskType) -> str:
        messages = [
            {"role": "system", "content": TASK_SYSTEMS[task_type]},
            {"role": "user", "content": _task_user_message(prompt, task_type)},
        ]
        return self._call_best_available(messages, task_type, _max_tokens_for(task_type))

    def _generate_code_ensemble_answer(self, prompt: str, task_type: TaskType) -> str:
        # Optional: two independent code candidates then a final selection. Off by default to
        # protect runtime and rate limits, but useful if the user wants maximum code accuracy.
        models = top_non_thinking_models(self.config.allowed_models, task_type)
        if len(models) < 2:
            return self._generate_answer(prompt, task_type)
        user = _task_user_message(prompt, task_type)
        a = self._chat_completion(models[0], [{"role": "system", "content": TASK_SYSTEMS[task_type]}, {"role": "user", "content": user}], _max_tokens_for(task_type))
        b = self._chat_completion(models[1], [{"role": "system", "content": TASK_SYSTEMS[task_type]}, {"role": "user", "content": user}], _max_tokens_for(task_type))
        select_messages = [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": f"Original task:\n{prompt}\n\nCandidate A:\n{a}\n\nCandidate B:\n{b}\n\nReturn the single best corrected final answer."},
        ]
        return self._call_best_available(select_messages, task_type, _verify_max_tokens_for(task_type))

    def _verify_answer(self, prompt: str, draft: str, task_type: TaskType) -> str:
        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": f"Original task:\n{prompt}\n\nDraft answer:\n{draft}\n\nReturn the corrected final answer only."},
        ]
        return self._call_best_available(messages, task_type, _verify_max_tokens_for(task_type))

    def _call_best_available(self, messages: list[dict[str, str]], task_type: TaskType, max_tokens: int) -> str:
        candidates = ranked_models(self.config.allowed_models, task_type)
        if not candidates:
            raise RuntimeError("No usable models found in ALLOWED_MODELS")

        # Try stable non-thinking models before explicit reasoning models to avoid final-answer truncation.
        stable = top_non_thinking_models(self.config.allowed_models, task_type)
        reasoning = [m for m in candidates if m not in stable]
        candidates = stable + reasoning

        last_error: Exception | None = None
        max_attempts = max(1, min(len(candidates), self.config.max_retries + 1))
        for i, model in enumerate(candidates[:max_attempts], start=1):
            try:
                budget = max_tokens
                if is_thinking_model(model):
                    budget = min(max(max_tokens, 2400), 3600)
                    messages_to_send = _add_no_think_instruction(messages)
                else:
                    messages_to_send = messages
                content = self._chat_completion(model=model, messages=messages_to_send, max_tokens=budget)
                content = _remove_thinking(content).strip()
                if content:
                    return content
                last_error = RuntimeError(f"Empty content from {model}")
            except Exception as exc:
                last_error = exc
                time.sleep(min(0.7 * i, 2.5))
        raise RuntimeError(f"All model attempts failed. Last error: {last_error}")

    def _chat_completion(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        url = _chat_completions_url(self.config.base_url)
        payload = {
            "model": model,
            "messages": messages,
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


def _add_no_think_instruction(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out = [dict(m) for m in messages]
    # Qwen/DeepSeek reasoning models often obey /no_think and final-only reminders.
    if out and out[-1].get("role") == "user":
        out[-1]["content"] = str(out[-1].get("content", "")) + "\n\n/no_think\nReturn final answer only."
    return out


def _task_user_message(prompt: str, task_type: TaskType) -> str:
    return (
        f"Detected task type: {task_type.value}. The detection may be wrong; ignore it if the original task says otherwise.\n\n"
        f"Original task:\n{prompt}"
    )


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _max_tokens_for(task_type: TaskType) -> int:
    return {
        TaskType.FACTUAL: 650,
        TaskType.MATH: 1200,
        TaskType.SENTIMENT: 220,
        TaskType.SUMMARY: 500,
        TaskType.NER: 700,
        TaskType.CODE_DEBUG: 1800,
        TaskType.LOGIC: 1400,
        TaskType.CODE_GEN: 2400,
    }[task_type]


def _verify_max_tokens_for(task_type: TaskType) -> int:
    return {
        TaskType.FACTUAL: 500,
        TaskType.MATH: 1000,
        TaskType.SENTIMENT: 180,
        TaskType.SUMMARY: 400,
        TaskType.NER: 600,
        TaskType.CODE_DEBUG: 1800,
        TaskType.LOGIC: 1200,
        TaskType.CODE_GEN: 2400,
    }[task_type]


def _clean_answer(answer: str, task_type: TaskType, prompt: str = "") -> str:
    text = _remove_thinking(answer).strip()
    text = re.sub(r"^\s*(final answer|answer|result)\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    # Preserve code formatting but remove a single surrounding markdown fence for code-generation tasks.
    if task_type == TaskType.CODE_GEN:
        text = _strip_single_code_fence(text)
        text = re.sub(r"^(?:here is|here's)\s+(?:the\s+)?(?:correct\s+)?(?:code|function)\s*:?\s*", "", text, flags=re.I).strip()

    # For code debug, keep explanatory line if present; only unwrap if the entire answer is one fence.
    if task_type == TaskType.CODE_DEBUG:
        text = _strip_single_code_fence(text)

    return text.strip()


def _remove_thinking(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()
    if text.lower().startswith("<think>"):
        m = re.search(r"(?:final answer|answer)\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    # Remove common reasoning headers if the final answer follows them.
    m = re.search(r"(?:final answer|therefore,? the answer is)\s*:?\s*(.+)$", text, flags=re.I | re.S)
    if m and len(m.group(1).strip()) > 0:
        return m.group(1).strip()
    return text


def _strip_single_code_fence(text: str) -> str:
    m = re.fullmatch(r"\s*```(?:[a-zA-Z0-9_+\-.#]*)?\s*\n(?P<code>.*?)\n```\s*", text, flags=re.DOTALL)
    if m:
        return m.group("code").strip()
    return text
