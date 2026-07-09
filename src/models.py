"""Model selection from ALLOWED_MODELS for Track 1.

No model IDs are hardcoded for final use; this module only ranks the model IDs
provided by the judging harness. v6 prioritizes stable instruction/code models
first and keeps explicit reasoning models as fallbacks because some reasoning
models may spend too much budget before producing a final answer.
"""
from __future__ import annotations

import re
from .classifier import TaskType

_BAD_MODEL_HINTS = (
    "embedding", "embed", "rerank", "whisper", "audio", "tts", "image", "vision",
    "moderation", "guard", "clip", "stable-diffusion", "diffusion", "flux",
)
_THINKING_HINTS = ("deepseek-r1", "/r1", "-r1", "reasoning", "reasoner", "thinking")


def parse_allowed_models(raw: str) -> list[str]:
    models = [m.strip() for m in raw.replace("\n", ",").split(",") if m.strip()]
    return list(dict.fromkeys(models))


def ranked_models(models: list[str], task_type: TaskType) -> list[str]:
    usable = [m for m in models if not any(bad in m.lower() for bad in _BAD_MODEL_HINTS)] or models[:]
    return sorted(usable, key=lambda m: _score_model(m, task_type), reverse=True)


def top_non_thinking_models(models: list[str], task_type: TaskType) -> list[str]:
    ranked = ranked_models(models, task_type)
    non = [m for m in ranked if not is_thinking_model(m)]
    return non or ranked


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

    # Instruction/chat tuning is much more important than raw model family.
    for hint, value in [
        ("instruct", 160), ("chat", 120), ("turbo", 35), ("preview", 0),
        ("base", -260), ("pretrain", -260), ("draft", -90),
    ]:
        if hint in m:
            score += value

    size_b = _largest_b_size(m)
    if size_b:
        score += min(size_b, 400) / 2.0

    # Strong current families. This is only ranking names supplied in ALLOWED_MODELS.
    family_bonus = [
        ("qwen3-coder", 210), ("qwen2.5-coder", 195), ("qwen2p5-coder", 195),
        ("kimi-k2", 190), ("deepseek-v3", 182), ("gpt-oss-120b", 175),
        ("gpt-oss", 145), ("qwen3", 145), ("qwen2.5", 130), ("qwen2p5", 130),
        ("qwen", 105), ("llama-v3.3", 112), ("llama-v3p3", 112),
        ("llama-4", 112), ("llama", 88), ("mistral-large", 88), ("mistral", 65),
        ("deepseek-r1", 75), ("deepseek", 98), ("mixtral", 65), ("gemma", 55),
    ]
    for hint, value in family_bonus:
        if hint in m:
            score += value

    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}:
        if "coder" in m or "code" in m:
            score += 260
        if "qwen" in m:
            score += 115
        if "kimi" in m:
            score += 100
        if "deepseek" in m:
            score += 95
    elif task_type in {TaskType.MATH, TaskType.LOGIC}:
        if "qwen" in m:
            score += 130
        if "deepseek-v3" in m:
            score += 115
        if "gpt-oss" in m:
            score += 105
        if "kimi" in m:
            score += 92
        if "llama" in m:
            score += 70
    elif task_type in {TaskType.SUMMARY, TaskType.NER, TaskType.SENTIMENT}:
        if "gpt-oss" in m:
            score += 115
        if "llama" in m:
            score += 105
        if "qwen" in m:
            score += 95
        if "kimi" in m:
            score += 85
    else:
        if "gpt-oss" in m:
            score += 120
        if "qwen" in m:
            score += 100
        if "llama" in m:
            score += 96
        if "deepseek" in m:
            score += 90
        if "kimi" in m:
            score += 86

    # Reasoning models can be great, but are risky with strict final-answer output.
    # Keep them as fallbacks unless they are the only strong option.
    if is_thinking_model(m):
        score -= 80
        if task_type in {TaskType.MATH, TaskType.LOGIC}:
            score += 25
    return score


def is_thinking_model(model_id: str) -> bool:
    m = model_id.lower()
    return any(h in m for h in _THINKING_HINTS)
