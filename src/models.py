"""Dynamic model selection from ALLOWED_MODELS.

The judging harness injects ALLOWED_MODELS. This module never invents a model id.
For v5, correctness is the priority, so large/reasoning-capable models are not
penalized by default. We still keep non-text models out.
"""

from __future__ import annotations

import re
from .classifier import TaskType

_BAD_MODEL_HINTS = (
    "embedding", "embed", "rerank", "whisper", "audio", "tts", "image", "vision",
    "moderation", "guard", "clip", "stable-diffusion", "diffusion", "flux",
)
_THINKING_HINTS = ("deepseek-r1", "r1", "reasoning", "reasoner", "thinking")


def parse_allowed_models(raw: str) -> list[str]:
    models = [m.strip() for m in raw.replace("\n", ",").split(",") if m.strip()]
    return list(dict.fromkeys(models))


def ranked_models(models: list[str], task_type: TaskType) -> list[str]:
    if not models:
        return []
    usable = [m for m in models if not any(bad in m.lower() for bad in _BAD_MODEL_HINTS)]
    if not usable:
        usable = models[:]
    return sorted(usable, key=lambda m: _score_model(m, task_type), reverse=True)


def _largest_b_size(model_id: str) -> float:
    vals: list[float] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*b", model_id.lower()):
        try:
            vals.append(float(match.group(1)))
        except ValueError:
            pass
    return max(vals) if vals else 0.0


def _score_model(model_id: str, task_type: TaskType) -> float:
    m = model_id.lower()
    score = 0.0

    # Prefer chat/instruction tuned models and larger models.
    for hint, value in [
        ("instruct", 90), ("chat", 70), ("turbo", 15), ("base", -160), ("pretrain", -160),
        ("draft", -50), ("preview", -5),
    ]:
        if hint in m:
            score += value

    size_b = _largest_b_size(m)
    if size_b:
        score += min(size_b, 500) / 2.2

    family_bonus = {
        "gpt-oss-120b": 170,
        "gpt-oss": 130,
        "qwen3-coder": 145,
        "qwen2.5-coder": 128,
        "qwen2p5-coder": 128,
        "qwen3": 128,
        "qwen2.5": 112,
        "qwen2p5": 112,
        "qwen": 92,
        "deepseek-r1": 125,
        "deepseek-v3": 124,
        "deepseek": 95,
        "llama-4": 105,
        "llama-v4": 105,
        "llama-v3.3": 92,
        "llama-v3p3": 92,
        "llama-v3.1": 82,
        "llama-v3p1": 82,
        "llama": 70,
        "kimi-k2": 120,
        "kimi": 92,
        "mixtral": 62,
        "mistral-large": 78,
        "mistral": 58,
        "gemma": 55,
    }
    for hint, value in family_bonus.items():
        if hint in m:
            score += value

    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}:
        if "coder" in m or "code" in m:
            score += 180
        if "qwen" in m:
            score += 82
        if "deepseek" in m:
            score += 72
        if "gpt-oss" in m:
            score += 55
    elif task_type in {TaskType.MATH, TaskType.LOGIC}:
        if any(h in m for h in _THINKING_HINTS):
            score += 90
        if "qwen" in m:
            score += 92
        if "gpt-oss" in m:
            score += 86
        if "deepseek" in m:
            score += 80
        if "llama" in m:
            score += 48
    elif task_type in {TaskType.NER, TaskType.SUMMARY, TaskType.SENTIMENT}:
        if "gpt-oss" in m:
            score += 90
        if "llama" in m:
            score += 72
        if "qwen" in m:
            score += 68
        if "deepseek" in m:
            score += 56
    else:
        if "gpt-oss" in m:
            score += 88
        if "qwen" in m:
            score += 72
        if "deepseek" in m:
            score += 70
        if "llama" in m:
            score += 66

    return score


def is_thinking_model(model_id: str) -> bool:
    m = model_id.lower()
    return any(h in m for h in _THINKING_HINTS)
