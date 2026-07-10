from __future__ import annotations

import os
import re
from .classifier import TaskType

def parse_allowed_models(raw: str) -> list[str]:
    return [m.strip() for m in raw.replace('\n', ',').split(',') if m.strip()]

def is_reasoning_model(model_id: str) -> bool:
    m = model_id.lower()
    return any(x in m for x in ['r1', 'reason', 'thinking', 'qwq', 'o1', 'o3'])

def ranked_models(models: list[str], task_type: TaskType | None = None) -> list[str]:
    # Default: trust harness order if requested. It is often curated by organizers.
    if os.getenv('MODEL_ORDER', 'hybrid').lower() == 'harness':
        return models[:]
    indexed = {m: i for i, m in enumerate(models)}
    return sorted(models, key=lambda m: (_score(m, task_type), -indexed.get(m, 0)), reverse=True)

def best_model(models: list[str], task_type: TaskType | None = None) -> str:
    ranked = ranked_models(models, task_type)
    if not ranked:
        raise RuntimeError('ALLOWED_MODELS is empty')
    return ranked[0]

def secondary_models(models: list[str], task_type: TaskType | None = None, limit: int = 2) -> list[str]:
    ranked = ranked_models(models, task_type)
    return ranked[:max(1, limit)]

def _b_size(model: str) -> float:
    vals = []
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*b', model.lower()):
        try: vals.append(float(m.group(1)))
        except Exception: pass
    return max(vals) if vals else 0.0

def _score(model: str, task_type: TaskType | None) -> float:
    m = model.lower()
    s = 0.0
    if any(x in m for x in ['embed', 'embedding', 'rerank', 'guard', 'whisper', 'audio', 'vision']):
        return -100000
    if 'instruct' in m: s += 300
    if 'chat' in m: s += 150
    if 'base' in m: s -= 1000
    s += min(_b_size(m), 405) * 2
    # Family priors for common Fireworks models.
    for hint, val in [
        ('gpt-oss-120b', 900), ('gpt-oss', 520), ('kimi-k2', 880), ('kimi', 560),
        ('qwen3-235', 860), ('qwen3-30', 520), ('qwen3', 620), ('qwen2.5-72', 620), ('qwen2p5-72', 620), ('qwen', 460),
        ('deepseek-v3', 800), ('deepseek-r1', 760), ('deepseek', 520),
        ('llama-v3p3-70b', 600), ('llama-3.3-70', 600), ('llama-v3.1-405', 760), ('llama', 360),
        ('mistral-large', 460), ('mixtral', 300), ('gemma', 260),
    ]:
        if hint in m: s += val
    if task_type in {TaskType.CODE_GEN, TaskType.CODE_DEBUG}:
        if any(x in m for x in ['coder', 'code']): s += 900
        if 'qwen' in m: s += 300
        if 'deepseek' in m: s += 260
        if is_reasoning_model(m): s += 60
    elif task_type in {TaskType.MATH, TaskType.LOGIC}:
        if is_reasoning_model(m): s += 420
        if 'qwen' in m: s += 260
        if 'deepseek' in m: s += 260
    else:
        if is_reasoning_model(m): s -= 160
    return s
