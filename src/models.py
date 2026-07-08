"""Dynamic model selection from ALLOWED_MODELS.

Model IDs are injected by the judging harness. This module never introduces a
model outside ALLOWED_MODELS; it only ranks the IDs that are provided.
"""

from __future__ import annotations

import re
from .classifier import TaskType

_BAD_MODEL_HINTS = (
    "embedding", "embed", "rerank", "whisper", "audio", "tts", "image", "vision",
    "moderation", "guard", "clip", "stable-diffusion", "diffusion", "flux",
)


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

    for hint, value in [
        ("instruct", 35), ("chat", 28), ("turbo", 8), ("base", -90), ("pretrain", -90),
        ("draft", -20), ("preview", -8),
    ]:
        if hint in m:
            score += value

    size_b = _largest_b_size(m)
    if size_b:
        score += min(size_b, 500) / 3.5

    family_bonus = {
        "gpt-oss-120b": 75,
        "gpt-oss": 55,
        "deepseek-r1": 82,
        "deepseek-v3": 70,
        "deepseek": 54,
        "qwen3-coder": 78,
        "qwen3": 68,
        "qwen2.5-coder": 64,
        "qwen2p5-coder": 64,
        "qwen2.5": 50,
        "qwen2p5": 50,
        "qwen": 42,
        "llama-4": 52,
        "llama-v4": 52,
        "llama-v3.3": 44,
        "llama-v3p3": 44,
        "llama-v3.1": 38,
        "llama-v3p1": 38,
        "llama": 28,
        "kimi": 48,
        "mixtral": 24,
        "mistral": 22,
    }
    for hint, value in family_bonus.items():
        if hint in m:
            score += value

    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}:
        if "coder" in m or "code" in m:
            score += 90
        if "deepseek" in m:
            score += 45
        if "qwen" in m:
            score += 45
        if "gpt-oss" in m:
            score += 28
    elif task_type in {TaskType.MATH, TaskType.LOGIC}:
        if any(x in m for x in ["r1", "reason", "thinking", "math"]):
            score += 90
        if "qwen" in m:
            score += 46
        if "deepseek" in m:
            score += 46
        if "gpt-oss" in m:
            score += 38
    elif task_type in {TaskType.NER, TaskType.SUMMARY, TaskType.SENTIMENT}:
        # Instruction-following quality is more important than raw reasoning here.
        if "gpt-oss" in m:
            score += 45
        if "llama" in m:
            score += 34
        if "qwen" in m:
            score += 32
        if "deepseek" in m:
            score += 24
    else:
        if "gpt-oss" in m:
            score += 48
        if "deepseek" in m:
            score += 36
        if "qwen" in m:
            score += 34
        if "llama" in m:
            score += 30

    return score
