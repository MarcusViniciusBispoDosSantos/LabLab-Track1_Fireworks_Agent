"""Robust model selection for ALLOWED_MODELS.

The harness supplies allowed model IDs. We never hardcode a model for execution;
we only rank the supplied IDs. v11 deliberately trusts harness order more than
previous versions because organizers often list the strongest/default model first.
"""
from __future__ import annotations

import os
import re
from .classifier import TaskType

_BAD_MODEL_HINTS = (
    "embedding", "embed", "rerank", "whisper", "audio", "tts", "image", "vision",
    "moderation", "guard", "clip", "stable-diffusion", "diffusion", "flux", "sdxl",
)
_THINKING_HINTS = ("deepseek-r1", "/r1", "-r1", "reasoning", "reasoner", "thinking", "qwq", "qwen3")


def parse_allowed_models(raw: str) -> list[str]:
    models = [m.strip() for m in raw.replace("\n", ",").split(",") if m.strip()]
    return list(dict.fromkeys(models))


def usable_models(models: list[str]) -> list[str]:
    usable = [m for m in models if not any(bad in m.lower() for bad in _BAD_MODEL_HINTS)]
    return usable or models[:]


def ranked_models(models: list[str], task_type: TaskType) -> list[str]:
    forced = os.getenv("FORCE_MODEL", "").strip()
    usable = usable_models(models)
    if forced and forced in usable:
        return [forced] + [m for m in usable if m != forced]

    strategy = os.getenv("MODEL_ORDER_STRATEGY", "hybrid").strip().lower()
    if strategy == "harness":
        return usable
    if strategy == "heuristic":
        return sorted(usable, key=lambda m: _score_model(m, task_type), reverse=True)

    # Hybrid: first harness model gets a large reliability bonus; then quality heuristic.
    order_index = {m: i for i, m in enumerate(usable)}
    return sorted(usable, key=lambda m: (_score_model(m, task_type) - order_index.get(m, 0) * 12 + (450 if order_index.get(m, 99) == 0 else 0)), reverse=True)


def primary_models(models: list[str], task_type: TaskType, limit: int = 3) -> list[str]:
    usable = usable_models(models)
    ranked = ranked_models(models, task_type)
    out: list[str] = []
    # Always try the first harness-provided model first unless FORCE_MODEL is set.
    forced = os.getenv("FORCE_MODEL", "").strip()
    if forced and forced in usable:
        out.append(forced)
    elif usable:
        out.append(usable[0])
    for model in ranked:
        if model not in out:
            out.append(model)
        if len(out) >= max(1, limit):
            break
    return out or ranked[:limit]


def selector_model(models: list[str], task_type: TaskType) -> str | None:
    ranked = ranked_models(models, task_type)
    return ranked[0] if ranked else None


def is_thinking_model(model_id: str) -> bool:
    m = model_id.lower()
    return any(h in m for h in _THINKING_HINTS)


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
    if any(bad in m for bad in _BAD_MODEL_HINTS): score -= 10000
    if "instruct" in m: score += 260
    if "chat" in m: score += 180
    if "base" in m or "pretrain" in m: score -= 800
    if "draft" in m or "spec" in m: score -= 200
    score += min(_largest_b_size(m), 405) * 1.0

    # Known high-quality Fireworks families seen in hackathon submissions.
    for hint, val in [
        ("kimi-k2p7-code", 650), ("kimi-k2", 540), ("kimi", 360),
        ("qwen3-235", 520), ("qwen3", 390), ("qwen2.5", 320), ("qwen2p5", 320),
        ("gpt-oss-120b", 460), ("gpt-oss", 350),
        ("deepseek-v3", 430), ("deepseek-r1", 390), ("deepseek", 300),
        ("llama-v3p3-70b", 350), ("llama-v3.3-70b", 350), ("llama", 220),
        ("mistral-large", 240), ("mistral", 150), ("gemma", 120),
    ]:
        if hint in m:
            score += val

    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}:
        if "code" in m or "coder" in m: score += 700
        if "kimi" in m: score += 320
        if "qwen" in m: score += 260
        if "deepseek" in m: score += 220
    elif task_type in {TaskType.MATH, TaskType.LOGIC}:
        if is_thinking_model(m): score += 360
        if "qwen" in m: score += 280
        if "deepseek" in m: score += 260
        if "kimi" in m: score += 210
    elif task_type in {TaskType.SUMMARY, TaskType.NER, TaskType.SENTIMENT}:
        if is_thinking_model(m): score -= 140
        if "gpt-oss" in m: score += 220
        if "llama" in m: score += 200
        if "qwen" in m: score += 190
        if "kimi" in m: score += 180
    else:
        if "gpt-oss" in m: score += 220
        if "llama" in m: score += 200
        if "qwen" in m: score += 190
        if "kimi" in m: score += 190
    return score
