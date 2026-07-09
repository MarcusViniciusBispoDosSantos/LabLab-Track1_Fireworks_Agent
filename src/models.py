"""Robust model selection from ALLOWED_MODELS.

No model ID is hardcoded for execution; this only ranks whatever the harness
provides. v8 mixes harness order with name-based quality heuristics because the
allowed list may already be ordered by the organizers.
"""
from __future__ import annotations

import os
import re
from .classifier import TaskType

_BAD_MODEL_HINTS = (
    "embedding", "embed", "rerank", "whisper", "audio", "tts", "image", "vision",
    "moderation", "guard", "clip", "stable-diffusion", "diffusion", "flux", "sdxl",
)
_THINKING_HINTS = ("deepseek-r1", "/r1", "-r1", "reasoning", "reasoner", "thinking", "qwq")


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
    # Preserve harness order as a meaningful signal: earlier models get a small bonus.
    order_index = {m: i for i, m in enumerate(usable)}
    return sorted(usable, key=lambda m: (_score_model(m, task_type) - order_index.get(m, 0) * 7), reverse=True)


def primary_models(models: list[str], task_type: TaskType, limit: int = 3) -> list[str]:
    """Return a small set of strong candidates.

    Includes both the best ranked model and the first harness-provided usable
    model when different. This protects against inaccurate name heuristics.
    """
    usable = usable_models(models)
    ranked = ranked_models(models, task_type)
    out: list[str] = []
    if ranked:
        out.append(ranked[0])
    if usable and usable[0] not in out:
        out.append(usable[0])
    for model in ranked:
        if model not in out:
            # Avoid three almost identical variants if possible.
            if _family(model) not in {_family(x) for x in out} or len(out) < 2:
                out.append(model)
        if len(out) >= limit:
            break
    for model in ranked:
        if len(out) >= limit:
            break
        if model not in out:
            out.append(model)
    return out


def verifier_model(models: list[str], task_type: TaskType) -> str | None:
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


def _family(model_id: str) -> str:
    m = model_id.lower()
    for fam in ["kimi", "deepseek-r1", "deepseek", "qwen", "gpt-oss", "llama", "mistral", "mixtral", "gemma"]:
        if fam in m:
            return fam
    last = m.split("/")[-1]
    return re.sub(r"[-_]?\d+(?:\.\d+)?b.*", "", last) or last


def _score_model(model_id: str, task_type: TaskType) -> float:
    m = model_id.lower()
    score = 0.0

    if any(bad in m for bad in _BAD_MODEL_HINTS): score -= 10000
    if "instruct" in m: score += 220
    if "chat" in m: score += 150
    if "turbo" in m: score += 40
    if "base" in m or "pretrain" in m: score -= 500
    if "draft" in m or "spec" in m: score -= 180

    size_b = _largest_b_size(m)
    if size_b:
        score += min(size_b, 405) * 0.8

    bonuses = [
        ("kimi-k2", 330), ("deepseek-v3", 310), ("qwen3-235", 305), ("qwen3", 260),
        ("gpt-oss-120b", 285), ("gpt-oss", 225),
        ("qwen2.5", 220), ("qwen2p5", 220),
        ("llama-4", 210), ("llama-v3.3", 195), ("llama-v3p3", 195), ("llama", 145),
        ("mistral-large", 150), ("mistral", 95), ("mixtral", 90), ("gemma", 80),
        ("deepseek-r1", 230), ("deepseek", 185),
    ]
    for hint, val in bonuses:
        if hint in m:
            score += val

    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}:
        if "coder" in m or "code" in m: score += 420
        if "qwen" in m: score += 170
        if "deepseek" in m: score += 150
        if "kimi" in m: score += 140
    elif task_type in {TaskType.MATH, TaskType.LOGIC}:
        if is_thinking_model(m): score += 270
        if "qwen" in m: score += 180
        if "deepseek" in m: score += 175
        if "gpt-oss" in m: score += 130
        if "kimi" in m: score += 120
    elif task_type in {TaskType.NER, TaskType.SUMMARY, TaskType.SENTIMENT}:
        # These usually benefit from instruction-following stability over long reasoning.
        if "llama" in m: score += 160
        if "gpt-oss" in m: score += 155
        if "qwen" in m: score += 140
        if "kimi" in m: score += 130
        if is_thinking_model(m): score -= 120
    else:
        if "gpt-oss" in m: score += 160
        if "llama" in m: score += 150
        if "qwen" in m: score += 140
        if "kimi" in m: score += 135
        if "deepseek" in m: score += 120
    return score
