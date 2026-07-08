"""Dynamic model selection from ALLOWED_MODELS.

The judging harness injects ALLOWED_MODELS. This module never invents a model id;
it only ranks the model IDs that are provided. v4 prioritizes instruction-tuned
non-reasoning chat models first because reasoning models can spend their token
budget in hidden <think> output and hurt benchmark accuracy.
"""

from __future__ import annotations

import os
import re
from .classifier import TaskType

_BAD_MODEL_HINTS = (
    "embedding", "embed", "rerank", "whisper", "audio", "tts", "image", "vision",
    "moderation", "guard", "clip", "stable-diffusion", "diffusion", "flux",
)
_THINKING_HINTS = ("deepseek-r1", "r1", "reasoning", "reasoner", "thinking", "o1", "o3")


def parse_allowed_models(raw: str) -> list[str]:
    models = [m.strip() for m in raw.replace("\n", ",").split(",") if m.strip()]
    return list(dict.fromkeys(models))


def ranked_models(models: list[str], task_type: TaskType) -> list[str]:
    if not models:
        return []
    usable = [m for m in models if not any(bad in m.lower() for bad in _BAD_MODEL_HINTS)]
    if not usable:
        usable = models[:]

    allow_thinking = os.getenv("ALLOW_THINKING_MODELS", "0").strip().lower() in {"1", "true", "yes", "on"}
    # Keep thinking models as fallback but do not use them first by default.
    return sorted(usable, key=lambda m: _score_model(m, task_type, allow_thinking), reverse=True)


def _largest_b_size(model_id: str) -> float:
    vals: list[float] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*b", model_id.lower()):
        try:
            vals.append(float(match.group(1)))
        except ValueError:
            pass
    return max(vals) if vals else 0.0


def _score_model(model_id: str, task_type: TaskType, allow_thinking: bool = False) -> float:
    m = model_id.lower()
    score = 0.0

    # Prefer instruction/chat models. Avoid base/pretrain and non-chat variants.
    for hint, value in [
        ("instruct", 60), ("chat", 48), ("turbo", 12), ("base", -140), ("pretrain", -140),
        ("draft", -30), ("preview", -5), ("embedding", -500),
    ]:
        if hint in m:
            score += value

    size_b = _largest_b_size(m)
    if size_b:
        score += min(size_b, 500) / 3.0

    # Family quality priors. These are only applied to IDs provided by ALLOWED_MODELS.
    family_bonus = {
        "gpt-oss-120b": 130,
        "gpt-oss": 92,
        "qwen3-coder": 108,
        "qwen2.5-coder": 94,
        "qwen2p5-coder": 94,
        "qwen3": 96,
        "qwen2.5": 78,
        "qwen2p5": 78,
        "qwen": 62,
        "deepseek-v3": 88,
        "deepseek": 58,
        "llama-4": 78,
        "llama-v4": 78,
        "llama-v3.3": 64,
        "llama-v3p3": 64,
        "llama-v3.1": 56,
        "llama-v3p1": 56,
        "llama": 44,
        "kimi": 66,
        "mixtral": 34,
        "mistral": 32,
        "gemma": 42,
    }
    for hint, value in family_bonus.items():
        if hint in m:
            score += value

    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}:
        if "coder" in m or "code" in m:
            score += 130
        if "qwen" in m:
            score += 60
        if "deepseek" in m and "r1" not in m:
            score += 45
        if "gpt-oss" in m:
            score += 42
    elif task_type in {TaskType.MATH, TaskType.LOGIC}:
        if "qwen" in m:
            score += 72
        if "gpt-oss" in m:
            score += 70
        if "deepseek-v3" in m:
            score += 56
        if "llama" in m:
            score += 38
    elif task_type in {TaskType.NER, TaskType.SUMMARY, TaskType.SENTIMENT}:
        if "gpt-oss" in m:
            score += 78
        if "llama" in m:
            score += 58
        if "qwen" in m:
            score += 54
        if "deepseek-v3" in m:
            score += 42
    else:
        if "gpt-oss" in m:
            score += 72
        if "qwen" in m:
            score += 58
        if "llama" in m:
            score += 54
        if "deepseek-v3" in m:
            score += 52

    is_thinking = any(h in m for h in _THINKING_HINTS)
    if is_thinking:
        # Reasoning models can be accurate, but they often waste max_tokens on hidden chain-of-thought.
        # Use them as fallback only unless explicitly enabled.
        score += 45 if allow_thinking else -260

    return score
