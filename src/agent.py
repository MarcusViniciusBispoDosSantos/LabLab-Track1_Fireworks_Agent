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
from .prompts import REFORMAT_SYSTEM, TASK_SYSTEMS, VERIFIER_SYSTEM
from .solvers import try_solve_locally


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str
    allowed_models: list[str]
    request_timeout_seconds: float = 28.0
    max_retries: int = 2
    verify_mode: str = "all"  # none | hard | all
    local_fast_paths: bool = True

    @classmethod
    def from_env(cls) -> "AgentConfig":
        required = ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))

        timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "28"))
        retries = int(os.getenv("MAX_RETRIES", "2"))
        legacy_verify = os.getenv("VERIFY_HARD_TASKS", "").strip().lower()
        verify_mode = os.getenv("VERIFY_MODE", "all").strip().lower()
        if legacy_verify in {"0", "false", "no", "off"} and "VERIFY_MODE" not in os.environ:
            verify_mode = "none"
        if verify_mode not in {"none", "hard", "all"}:
            verify_mode = "all"
        local_fast_paths = os.getenv("ENABLE_LOCAL_FAST_PATHS", "1").strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"].rstrip("/"),
            allowed_models=parse_allowed_models(os.environ["ALLOWED_MODELS"]),
            request_timeout_seconds=timeout,
            max_retries=max(0, retries),
            verify_mode=verify_mode,
            local_fast_paths=local_fast_paths,
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
                return {"task_id": task_id, "answer": _clean_answer(local, task_type)}

        answer = self._generate_answer(prompt, task_type)

        if self._should_verify(task_type):
            try:
                answer = self._verify_answer(prompt, answer, task_type)
            except Exception:
                # Keeping a generated answer is better than returning a failure string.
                pass

        # A very light final format pass helps with verbose models, while avoiding a
        # third model call for normal cases.
        answer = _clean_answer(answer, task_type)
        return {"task_id": task_id, "answer": answer}

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

    def _verify_answer(self, prompt: str, draft: str, task_type: TaskType) -> str:
        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": f"Original task category: {task_type.value}\n\nOriginal task:\n{prompt}\n\nDraft answer:\n{draft}"},
        ]
        return self._call_best_available(messages, task_type, _verify_max_tokens_for(task_type))

    def _reformat_answer(self, prompt: str, draft: str, task_type: TaskType) -> str:
        messages = [
            {"role": "system", "content": REFORMAT_SYSTEM},
            {"role": "user", "content": f"Original task:\n{prompt}\n\nAnswer to format:\n{draft}"},
        ]
        return self._call_best_available(messages, task_type, _verify_max_tokens_for(task_type))

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
        max_attempts = max(1, min(len(candidates), self.config.max_retries + 1))

        for model in candidates[:max_attempts]:
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
                time.sleep(min(0.6 * attempts, 1.8))

        raise RuntimeError(f"All model attempts failed. Last error: {last_error}")

    def _chat_completion(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        """Call Fireworks through FIREWORKS_BASE_URL using only Python stdlib."""
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
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        if isinstance(item.get("text"), str):
                            parts.append(item["text"])
                        elif item.get("type") == "text" and isinstance(item.get("content"), str):
                            parts.append(item["content"])
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    return "\n".join(parts)
        text = first.get("text")
        if isinstance(text, str):
            return text
        raise RuntimeError(f"Could not parse assistant content: {raw[:700]}")


def _task_user_message(prompt: str, task_type: TaskType) -> str:
    return (
        f"Task category: {task_type.value}\n"
        "Solve the original task below. Follow its requested output format exactly.\n\n"
        f"Original task:\n{prompt}"
    )


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _max_tokens_for(task_type: TaskType) -> int:
    return {
        TaskType.FACTUAL: 650,
        TaskType.MATH: 750,
        TaskType.SENTIMENT: 220,
        TaskType.SUMMARY: 320,
        TaskType.NER: 520,
        TaskType.CODE_DEBUG: 1400,
        TaskType.LOGIC: 850,
        TaskType.CODE_GEN: 1700,
    }[task_type]


def _verify_max_tokens_for(task_type: TaskType) -> int:
    return {
        TaskType.FACTUAL: 650,
        TaskType.MATH: 700,
        TaskType.SENTIMENT: 220,
        TaskType.SUMMARY: 320,
        TaskType.NER: 520,
        TaskType.CODE_DEBUG: 1400,
        TaskType.LOGIC: 850,
        TaskType.CODE_GEN: 1700,
    }[task_type]


def _clean_answer(answer: str, task_type: TaskType) -> str:
    text = answer.strip()
    text = _remove_thinking(text)
    text = re.sub(r"^\s*(final answer|answer|result)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\s*Here is (?:the )?", "", text, flags=re.IGNORECASE).strip()

    if task_type == TaskType.CODE_GEN:
        text = _strip_single_code_fence(text)
        text = re.sub(
            r"^(?:the )?(?:Python|JavaScript|TypeScript|Java|C\+\+|SQL)?\s*code(?: is)?:?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
    return text.strip()


def _remove_thinking(text: str) -> str:
    # Reasoning models may emit <think>...</think>. The final answer usually follows.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if "</think>" in text.lower():
        # Remove anything before the last closing tag.
        parts = re.split(r"</think>", text, flags=re.IGNORECASE)
        text = parts[-1].strip()
    if text.lower().startswith("<think>"):
        # If a model emitted an unclosed think block, try to keep content after a blank line.
        chunks = re.split(r"\n\s*\n", text, maxsplit=1)
        text = chunks[1].strip() if len(chunks) > 1 else text
    return text


def _strip_single_code_fence(text: str) -> str:
    m = re.fullmatch(r"\s*```(?:[a-zA-Z0-9_+\-.#]*)?\s*\n(?P<code>.*?)\n```\s*", text, flags=re.DOTALL)
    if m:
        return m.group("code").strip()
    return text
