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
from .models import parse_allowed_models, ranked_models
from .prompts import TASK_SYSTEMS, VERIFIER_SYSTEM


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str
    allowed_models: list[str]
    request_timeout_seconds: float = 28.0
    max_retries: int = 2
    verify_hard_tasks: bool = False

    @classmethod
    def from_env(cls) -> "AgentConfig":
        required = ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))

        timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "28"))
        retries = int(os.getenv("MAX_RETRIES", "2"))
        verify = os.getenv("VERIFY_HARD_TASKS", "0").strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"].rstrip("/"),
            allowed_models=parse_allowed_models(os.environ["ALLOWED_MODELS"]),
            request_timeout_seconds=timeout,
            max_retries=max(0, retries),
            verify_hard_tasks=verify,
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
        answer = self._generate_answer(prompt, task_type)

        if self.config.verify_hard_tasks and task_type in {
            TaskType.MATH,
            TaskType.LOGIC,
            TaskType.CODE_DEBUG,
            TaskType.CODE_GEN,
        }:
            try:
                answer = self._verify_answer(prompt, answer, task_type)
            except Exception:
                # Keep the original answer if verification fails; malformed/missing output is worse.
                pass

        return {"task_id": task_id, "answer": _clean_answer(answer, task_type)}

    def _generate_answer(self, prompt: str, task_type: TaskType) -> str:
        messages = [
            {"role": "system", "content": TASK_SYSTEMS[task_type]},
            {"role": "user", "content": prompt},
        ]
        return self._call_best_available(messages, task_type, _max_tokens_for(task_type))

    def _verify_answer(self, prompt: str, draft: str, task_type: TaskType) -> str:
        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": f"Original task:\n{prompt}\n\nDraft answer:\n{draft}"},
        ]
        return self._call_best_available(messages, task_type, _max_tokens_for(task_type))

    def _call_best_available(
        self,
        messages: list[dict[str, str]],
        task_type: TaskType,
        max_tokens: int,
    ) -> str:
        candidates = ranked_models(self.config.allowed_models, task_type)
        if not candidates:
            raise RuntimeError("No usable models found in ALLOWED_MODELS")

        last_error: Exception | None = None
        attempts = 0
        max_attempts = max(1, self.config.max_retries + 1)

        # Try strongest-ranked allowed models first. Never use a model outside ALLOWED_MODELS.
        for model in candidates[: max(3, max_attempts)]:
            if attempts >= max_attempts:
                break
            attempts += 1
            try:
                content = self._chat_completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                if content.strip():
                    return content.strip()
                last_error = RuntimeError(f"Empty response from {model}")
            except Exception as exc:
                last_error = exc
                time.sleep(min(0.8 * attempts, 2.0))

        raise RuntimeError(f"All model attempts failed. Last error: {last_error}")

    def _chat_completion(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        """Call Fireworks through FIREWORKS_BASE_URL using only Python stdlib.

        The harness records token usage through this base URL, so every request must go
        through config.base_url. No model ID is hardcoded; `model` always comes from
        ALLOWED_MODELS.
        """
        url = _chat_completions_url(self.config.base_url)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
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
            raise RuntimeError(f"HTTP {exc.code} from Fireworks proxy: {body[:500]}") from exc

        parsed = json.loads(raw)
        choices = parsed.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in response: {raw[:500]}")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        text = first.get("text") if isinstance(first, dict) else None
        if isinstance(text, str):
            return text
        raise RuntimeError(f"Could not parse assistant content: {raw[:500]}")


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _max_tokens_for(task_type: TaskType) -> int:
    return {
        TaskType.FACTUAL: 550,
        TaskType.MATH: 550,
        TaskType.SENTIMENT: 180,
        TaskType.SUMMARY: 260,
        TaskType.NER: 380,
        TaskType.CODE_DEBUG: 1100,
        TaskType.LOGIC: 650,
        TaskType.CODE_GEN: 1400,
    }[task_type]


def _clean_answer(answer: str, task_type: TaskType) -> str:
    text = answer.strip()
    text = re.sub(r"^\s*(final answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    if task_type == TaskType.CODE_GEN:
        text = _strip_single_code_fence(text)
    return text.strip()


def _strip_single_code_fence(text: str) -> str:
    m = re.fullmatch(r"\s*```(?:[a-zA-Z0-9_+\-.#]*)?\s*\n(?P<code>.*?)\n```\s*", text, flags=re.DOTALL)
    if m:
        return m.group("code").strip()
    return text
