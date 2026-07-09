"""Model selection from ALLOWED_MODELS for Track 1.

The code never hardcodes model IDs for use. It only ranks whatever model IDs are
provided by the judging harness in ALLOWED_MODELS.
"""
from __future__ import annotations

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


def ranked_models(models: list[str], task_type: TaskType) -> list[str]:
    usable = [m for m in models if not any(bad in m.lower() for bad in _BAD_MODEL_HINTS)] or models[:]
    return sorted(usable, key=lambda m: _score_model(m, task_type), reverse=True)


def diverse_models(models: list[str], task_type: TaskType, limit: int = 3) -> list[str]:
    """Return a small, diverse set for ensemble solving.

    We avoid selecting three near-identical model IDs from one family when a
    different strong family is available.
    """
    ranked = ranked_models(models, task_type)
    out: list[str] = []
    seen_family: set[str] = set()
    for model in ranked:
        fam = _family(model)
        if fam not in seen_family:
            out.append(model)
            seen_family.add(fam)
        if len(out) >= limit:
            return out
    for model in ranked:
        if model not in out:
            out.append(model)
        if len(out) >= limit:
            break
    return out


def best_selector_model(models: list[str], task_type: TaskType) -> str | None:
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
    # fallback to last path segment without size/version suffixes
    last = m.split("/")[-1]
    return re.sub(r"[-_]?\d+(?:\.\d+)?b.*", "", last) or last


def _score_model(model_id: str, task_type: TaskType) -> float:
    m = model_id.lower()
    score = 0.0

    # Avoid non-chat/base models.
    if "instruct" in m: score += 180
    if "chat" in m: score += 120
    if "turbo" in m: score += 30
    if "base" in m or "pretrain" in m: score -= 320
    if "draft" in m or "spec" in m: score -= 120

    size_b = _largest_b_size(m)
    if size_b:
        score += min(size_b, 405) * 0.7

    # Strong current families. These are only name-based priorities among models
    # already allowed by the harness.
    family_bonus = [
        ("kimi-k2", 260), ("deepseek-v3", 245), ("qwen3-235", 240),
        ("qwen3", 210), ("qwen2.5", 175), ("qwen2p5", 175),
        ("gpt-oss-120b", 215), ("gpt-oss", 170),
        ("llama-4", 165), ("llama-v3.3", 155), ("llama-v3p3", 155), ("llama", 115),
        ("mistral-large", 120), ("mistral", 80), ("mixtral", 75), ("gemma", 65),
        ("deepseek-r1", 150), ("deepseek", 130),
    ]
    for hint, value in family_bonus:
        if hint in m:
            score += value

    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}:
        if "coder" in m or "code" in m: score += 320
        if "qwen" in m: score += 130
        if "kimi" in m: score += 120
        if "deepseek" in m: score += 110
    elif task_type in {TaskType.MATH, TaskType.LOGIC}:
        if is_thinking_model(m): score += 160
        if "qwen" in m: score += 150
        if "deepseek" in m: score += 150
        if "gpt-oss" in m: score += 95
        if "kimi" in m: score += 90
    elif task_type in {TaskType.SUMMARY, TaskType.NER, TaskType.SENTIMENT}:
        if "llama" in m: score += 120
        if "gpt-oss" in m: score += 110
        if "qwen" in m: score += 100
        if "kimi" in m: score += 95
    else:
        if "gpt-oss" in m: score += 120
        if "llama" in m: score += 115
        if "qwen" in m: score += 105
        if "kimi" in m: score += 100
        if "deepseek" in m: score += 95

    # For non-reasoning tasks, reasoning models often waste budget and can format poorly.
    if is_thinking_model(m) and task_type not in {TaskType.MATH, TaskType.LOGIC}:
        score -= 100
    return score
