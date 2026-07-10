"""Model selection for ALLOWED_MODELS.

No model ID is hardcoded for execution. We only rank the IDs injected by the
hackathon harness and then call one of those exact IDs.
"""
from __future__ import annotations

import json
import os
import re
from .classifier import TaskType

_BAD_MODEL_HINTS = (
    "embedding", "embed", "rerank", "whisper", "audio", "tts", "image", "vision",
    "moderation", "guard", "clip", "stable-diffusion", "diffusion", "flux", "sdxl",
)
_THINKING_HINTS = ("deepseek-r1", "/r1", "-r1", "reasoning", "reasoner", "thinking", "qwq", "qwen3")


def parse_allowed_models(raw: str) -> list[str]:
    raw = (raw or "").strip()
    models: list[str] = []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                models = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            models = []
    if not models:
        models = [m.strip().strip('"').strip("'") for m in raw.replace("\n", ",").split(",") if m.strip().strip('"').strip("'")]
    return list(dict.fromkeys(models))


def usable_models(models: list[str]) -> list[str]:
    usable = [m for m in models if not any(bad in m.lower() for bad in _BAD_MODEL_HINTS)]
    return usable or models[:]


def ranked_models(models: list[str], task_type: TaskType | None = None) -> list[str]:
    usable = usable_models(models)
    forced = os.getenv("FORCE_MODEL", "").strip()
    if forced and forced in usable:
        return [forced] + [m for m in usable if m != forced]

    strategy = os.getenv("MODEL_ORDER_STRATEGY", "hybrid").strip().lower()
    if strategy == "harness":
        return usable
    if strategy == "heuristic":
        return sorted(usable, key=lambda m: _score_model(m, task_type), reverse=True)

    # Hybrid: quality heuristic with a modest bonus for harness order.
    order_index = {m: i for i, m in enumerate(usable)}
    return sorted(
        usable,
        key=lambda m: (_score_model(m, task_type) - order_index.get(m, 0) * 8 + (120 if order_index.get(m, 99) == 0 else 0)),
        reverse=True,
    )


def primary_models(models: list[str], task_type: TaskType | None = None, limit: int = 3) -> list[str]:
    out: list[str] = []
    for m in ranked_models(models, task_type):
        if m not in out:
            out.append(m)
        if len(out) >= max(1, limit):
            break
    return out


def selector_model(models: list[str], task_type: TaskType | None = None) -> str | None:
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


def _score_model(model_id: str, task_type: TaskType | None = None) -> float:
    m = model_id.lower()
    score = 0.0
    if any(bad in m for bad in _BAD_MODEL_HINTS):
        score -= 10000
    if "instruct" in m: score += 250
    if "chat" in m: score += 160
    if "base" in m or "pretrain" in m: score -= 900
    if "draft" in m or "speculative" in m: score -= 400
    if "8b" in m or "7b" in m or "3b" in m or "1b" in m: score -= 120
    score += min(_largest_b_size(m), 405) * 1.2

    # Strong general model families commonly exposed by Fireworks.
    hints = [
        ("kimi-k2", 780), ("kimi", 420),
        ("qwen3-235", 760), ("qwen3-30", 430), ("qwen3", 510),
        ("qwen2.5-72", 470), ("qwen2p5-72", 470), ("qwen2.5", 330), ("qwen2p5", 330),
        ("gpt-oss-120b", 690), ("gpt-oss", 430),
        ("deepseek-v3", 650), ("deepseek-r1", 620), ("deepseek", 420),
        ("llama-v3p3-70b", 460), ("llama-3.3-70", 460), ("llama-v3.1-405", 620), ("llama", 240),
        ("mistral-large", 360), ("mixtral", 260), ("gemma", 180),
    ]
    for hint, val in hints:
        if hint in m:
            score += val

    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}:
        if "code" in m or "coder" in m: score += 750
        if "kimi" in m: score += 380
        if "qwen" in m: score += 340
        if "deepseek" in m: score += 310
        if "gpt-oss" in m: score += 170
    elif task_type in {TaskType.MATH, TaskType.LOGIC}:
        if is_thinking_model(m): score += 420
        if "qwen" in m: score += 360
        if "deepseek" in m: score += 340
        if "gpt-oss" in m: score += 260
        if "kimi" in m: score += 240
    elif task_type in {TaskType.SUMMARY, TaskType.NER, TaskType.SENTIMENT}:
        if is_thinking_model(m): score -= 100
        if "gpt-oss" in m: score += 290
        if "kimi" in m: score += 270
        if "qwen" in m: score += 250
        if "llama" in m: score += 220
    else:
        if "gpt-oss" in m: score += 300
        if "kimi" in m: score += 290
        if "qwen" in m: score += 270
        if "llama" in m: score += 210
    return score
