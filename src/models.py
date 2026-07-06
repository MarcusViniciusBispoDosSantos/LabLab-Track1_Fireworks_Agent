"""Dynamic model selection from ALLOWED_MODELS.

The allowed model IDs are published/injected at runtime, so this code never
hardcodes a specific model ID for submission. It ranks whatever IDs are provided.
"""

from __future__ import annotations

import re
from .classifier import TaskType

_BAD_MODEL_HINTS = (
    "embedding", "embed", "rerank", "whisper", "audio", "tts", "image", "vision",
    "moderation", "guard", "clip", "stable-diffusion",
)


def parse_allowed_models(raw: str) -> list[str]:
    models = [m.strip() for m in raw.split(",") if m.strip()]
    # Preserve order while deduplicating.
    return list(dict.fromkeys(models))


def ranked_models(models: list[str], task_type: TaskType) -> list[str]:
    if not models:
        return []
    usable = [m for m in models if not any(bad in m.lower() for bad in _BAD_MODEL_HINTS)]
    if not usable:
        usable = models[:]
    return sorted(usable, key=lambda m: _score_model(m, task_type), reverse=True)


def _score_model(model_id: str, task_type: TaskType) -> float:
    m = model_id.lower()
    score = 0.0

    # Prefer instruction/chat models over base models.
    for hint, value in [
        ("instruct", 18), ("chat", 14), ("turbo", 8), ("fast", 4),
        ("base", -40), ("pretrain", -40),
    ]:
        if hint in m:
            score += value

    # Approximate scale from model name: 405b, 120b, 72b, etc.
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*b", m):
        try:
            size_b = float(match.group(1))
            score += min(size_b, 500) / 5.0
        except ValueError:
            pass

    # MoE / family hints common in hosted LLM IDs.
    family_bonus = {
        "gpt-oss": 38,
        "deepseek-v3": 36,
        "deepseek": 30,
        "qwen3": 32,
        "qwen2.5": 28,
        "qwen2p5": 28,
        "qwen": 24,
        "llama-4": 30,
        "llama-v4": 30,
        "llama-v3.3": 28,
        "llama-v3p3": 28,
        "llama-v3.1": 26,
        "llama-v3p1": 26,
        "llama": 22,
        "mixtral": 18,
        "mistral": 16,
    }
    for hint, value in family_bonus.items():
        if hint in m:
            score += value

    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}:
        if "coder" in m or "code" in m:
            score += 55
        if "deepseek" in m:
            score += 30
        if "qwen" in m:
            score += 30
        if "gpt-oss" in m:
            score += 18
        if "llama" in m:
            score += 10
    elif task_type in {TaskType.MATH, TaskType.LOGIC}:
        if "r1" in m or "reason" in m:
            score += 45
        if "qwen" in m:
            score += 30
        if "deepseek" in m:
            score += 28
        if "gpt-oss" in m:
            score += 22
        if "llama" in m:
            score += 14
    elif task_type in {TaskType.SUMMARY, TaskType.SENTIMENT, TaskType.NER, TaskType.FACTUAL}:
        if "gpt-oss" in m:
            score += 28
        if "llama" in m:
            score += 24
        if "qwen" in m:
            score += 20
        if "deepseek" in m:
            score += 18

    return score
