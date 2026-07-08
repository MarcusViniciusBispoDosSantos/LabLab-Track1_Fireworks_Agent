"""Safe local fast paths for obvious Track 1 tasks.

These are not answer caches. They only solve simple general patterns with
standard deterministic logic. If a prompt is not clearly supported, the agent
falls back to Fireworks AI.
"""

from __future__ import annotations

import ast
import operator
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable

from .classifier import TaskType

_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "love", "loved", "fast", "clean", "helpful",
    "easy", "smooth", "perfect", "reliable", "happy", "satisfied", "impressive", "best",
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "hated", "slow", "broken", "buggy", "fails",
    "failed", "failure", "crash", "crashes", "error", "poor", "worst", "disappointed",
    "confusing", "unusable", "late", "expensive", "problem", "issue",
}


def try_solve_locally(prompt: str, task_type: TaskType) -> str | None:
    if task_type == TaskType.MATH:
        return _try_math(prompt)
    if task_type == TaskType.SENTIMENT:
        return _try_sentiment(prompt)
    if task_type == TaskType.CODE_DEBUG:
        return _try_simple_code_debug(prompt)
    return None


def _try_math(prompt: str) -> str | None:
    # Pure arithmetic expression, e.g. "What is 12 * (5 + 3)?"
    expr = _extract_arithmetic_expression(prompt)
    if expr:
        try:
            value = _safe_eval(expr)
            return _format_number(value)
        except Exception:
            pass

    # Common percentage sequence: "has 1200 users, grows 15%, loses 8%".
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", prompt.replace(",", ""))
    low = prompt.lower()
    if len(nums) >= 3 and any(k in low for k in ["grows", "growth", "increase", "increases"]) and any(k in low for k in ["loses", "loss", "decrease", "decreases", "drops"]):
        try:
            base = Decimal(nums[0])
            pct1 = Decimal(nums[1]) / Decimal(100)
            pct2 = Decimal(nums[2]) / Decimal(100)
            value = base * (Decimal(1) + pct1) * (Decimal(1) - pct2)
            return _format_decimal(value)
        except Exception:
            return None

    # "What is X percent of Y?"
    m = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*(?:percent|%)\s+of\s+(\d+(?:\.\d+)?)", low)
    if m:
        value = Decimal(m.group(1)) * Decimal(m.group(2)) / Decimal(100)
        return _format_decimal(value)

    return None


def _extract_arithmetic_expression(prompt: str) -> str | None:
    # Avoid grabbing code snippets.
    if "def " in prompt or "return " in prompt or "```" in prompt:
        return None
    candidates = re.findall(r"[0-9][0-9\s+\-*/().%^]*[0-9)]", prompt)
    candidates = [c.strip().replace("^", "**") for c in candidates if any(op in c for op in "+-*/^")]
    if not candidates:
        return None
    # Use the longest expression-like candidate.
    expr = max(candidates, key=len)
    if re.fullmatch(r"[0-9\s+\-*/().*]+", expr):
        return expr
    return None


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval")
    return float(_eval_ast(node.body))


def _eval_ast(node: ast.AST) -> float:
    ops: dict[type[ast.AST], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in ops:
        return ops[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    raise ValueError("unsupported expression")


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return (f"{value:.10f}".rstrip("0").rstrip("."))


def _format_decimal(value: Decimal) -> str:
    q = value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP).normalize()
    text = format(q, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _try_sentiment(prompt: str) -> str | None:
    # Only use a deterministic sentiment answer when the prompt explicitly asks for a label.
    low = prompt.lower()
    if not any(k in low for k in ["sentiment", "classify", "positive", "negative", "neutral", "polarity"]):
        return None

    # Analyze text after a colon when available; this avoids classifying the instructions.
    text = prompt.split(":", 1)[1] if ":" in prompt else prompt
    words = re.findall(r"[a-zA-Z']+", text.lower())
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)

    if pos and neg:
        label = "Mixed"
    elif pos:
        label = "Positive"
    elif neg:
        label = "Negative"
    else:
        label = "Neutral"

    if "justify" in low or "why" in low or label == "Mixed":
        if label == "Mixed":
            return "Mixed — it contains both positive and negative signals."
        return f"{label} — the wording is primarily {label.lower()}."
    return label


def _try_simple_code_debug(prompt: str) -> str | None:
    # General typo pattern: len(num) inside a function whose parameter is nums.
    if re.search(r"def\s+\w+\s*\(\s*nums\s*\)", prompt) and "len(num)" in prompt:
        code_match = re.search(r"(def\s+\w+\s*\(\s*nums\s*\):[\s\S]+)", prompt)
        if code_match:
            code = code_match.group(1).strip().replace("len(num)", "len(nums)")
            return "Bug: `num` is undefined; use `nums`.\n\n" + code
    return None
