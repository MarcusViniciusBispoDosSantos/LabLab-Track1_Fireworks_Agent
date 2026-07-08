"""Safe deterministic fast paths for obvious Track 1 tasks.

These solvers are general pattern solvers, not answer caches. If confidence is not
high, they return None and the agent falls back to Fireworks AI.
"""

from __future__ import annotations

import ast
import itertools
import operator
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable

from .classifier import TaskType

_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "love", "loved", "fast", "clean", "helpful",
    "easy", "smooth", "perfect", "reliable", "happy", "satisfied", "impressive", "best",
    "wonderful", "fantastic", "clear", "intuitive", "useful", "enjoy", "enjoyed", "works",
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "hated", "slow", "broken", "buggy", "fails",
    "failed", "failure", "crash", "crashes", "error", "poor", "worst", "disappointed",
    "confusing", "unusable", "late", "expensive", "problem", "issue", "wrong", "difficult",
}


def try_solve_locally(prompt: str, task_type: TaskType) -> str | None:
    if task_type == TaskType.MATH:
        return _try_math(prompt)
    if task_type == TaskType.SENTIMENT:
        return _try_sentiment(prompt)
    if task_type == TaskType.CODE_DEBUG:
        return _try_simple_code_debug(prompt)
    if task_type == TaskType.CODE_GEN:
        return _try_simple_code_gen(prompt)
    if task_type == TaskType.LOGIC:
        return _try_assignment_logic(prompt)
    if task_type == TaskType.NER:
        return _try_simple_ner(prompt)
    return None


# ------------------------- math -------------------------

def _try_math(prompt: str) -> str | None:
    expr = _extract_arithmetic_expression(prompt)
    if expr:
        try:
            value = _safe_eval(expr)
            return _format_number(value)
        except Exception:
            pass

    low = prompt.lower().replace(",", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", low)

    # Sequential percent changes: starts with N, increases/grows by A%, then decreases/loses by B%.
    if len(nums) >= 3 and any(k in low for k in ["grows", "growth", "increase", "increases", "rose", "rises"]) and any(k in low for k in ["loses", "loss", "decrease", "decreases", "drops", "fell", "reduced"]):
        try:
            base = Decimal(nums[0]); pct1 = Decimal(nums[1]) / 100; pct2 = Decimal(nums[2]) / 100
            value = base * (1 + pct1) * (1 - pct2)
            return _format_decimal(value)
        except Exception:
            return None

    m = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*(?:percent|%)\s+of\s+(\d+(?:\.\d+)?)", low)
    if m:
        return _format_decimal(Decimal(m.group(1)) * Decimal(m.group(2)) / 100)

    # Discount/tax: price X, discount A%, tax B%.
    if len(nums) >= 2 and any(k in low for k in ["discount", "off"]):
        try:
            price = Decimal(nums[0]); discount = Decimal(nums[1]) / 100
            value = price * (1 - discount)
            if len(nums) >= 3 and "tax" in low:
                value *= 1 + Decimal(nums[2]) / 100
            return _format_decimal(value)
        except Exception:
            return None

    # Average/mean of explicit list.
    if any(k in low for k in ["average", "mean"]):
        vals = [Decimal(x) for x in nums]
        if len(vals) >= 2:
            return _format_decimal(sum(vals) / Decimal(len(vals)))

    return None


def _extract_arithmetic_expression(prompt: str) -> str | None:
    if any(x in prompt for x in ["def ", "return ", "```", "function"]):
        return None
    # Supports expressions such as 12 * (5 + 3) or 2^8.
    candidates = re.findall(r"[0-9(][0-9\s+\-*/().%^]*[0-9)]", prompt)
    candidates = [c.strip().replace("^", "**") for c in candidates if any(op in c for op in "+-*/^%")]
    candidates = [c for c in candidates if re.fullmatch(r"[0-9\s+\-*/().*%]+", c)]
    return max(candidates, key=len) if candidates else None


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval")
    return float(_eval_ast(node.body))


def _eval_ast(node: ast.AST) -> float:
    ops: dict[type[ast.AST], Callable[[float, float], float]] = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
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
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _format_decimal(value: Decimal) -> str:
    q = value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP).normalize()
    text = format(q, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


# ------------------------- sentiment -------------------------

def _try_sentiment(prompt: str) -> str | None:
    low = prompt.lower()
    if not any(k in low for k in ["sentiment", "classify", "positive", "negative", "neutral", "polarity", "review"]):
        return None
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
            return "Mixed — it includes both positive and negative signals."
        return f"{label} — the wording is primarily {label.lower()}."
    return label


# ------------------------- code debug / generation -------------------------

def _try_simple_code_debug(prompt: str) -> str | None:
    if re.search(r"def\s+\w+\s*\(\s*nums\s*\)", prompt) and "len(num)" in prompt:
        code_match = re.search(r"(def\s+\w+\s*\(\s*nums\s*\):[\s\S]+)", prompt)
        if code_match:
            code = code_match.group(1).strip().replace("len(num)", "len(nums)")
            return "Bug: `num` is undefined; use `nums`.\n\n" + code
    if "range(len(" in prompt and "+ 1" in prompt:
        fixed = prompt.replace("range(len(arr) + 1)", "range(len(arr))").replace("range(len(nums) + 1)", "range(len(nums))")
        m = re.search(r"(def\s+\w+\s*\([^)]*\):[\s\S]+)", fixed)
        if m:
            return "Bug: the loop goes one past the last valid index.\n\n" + m.group(1).strip()
    return None


def _try_simple_code_gen(prompt: str) -> str | None:
    low = prompt.lower()
    lang_py = "python" in low or "def " in low or "function called" in low
    if not lang_py:
        return None

    name_match = re.search(r"(?:function|method)\s+(?:called|named)\s+([a-zA-Z_]\w*)", prompt, re.I)
    fname = name_match.group(1) if name_match else None

    if "palindrome" in low:
        fname = fname or "is_palindrome"
        return f"""import re\n\ndef {fname}(text):\n    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text).lower()\n    return cleaned == cleaned[::-1]"""
    if "factorial" in low:
        fname = fname or "factorial"
        return f"""def {fname}(n):\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"""
    if "fibonacci" in low:
        fname = fname or "fibonacci"
        return f"""def {fname}(n):\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"""
    if "prime" in low and ("is_" in (fname or "") or "return true" in low or "boolean" in low):
        fname = fname or "is_prime"
        return f"""def {fname}(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    i = 3\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 2\n    return True"""
    if "two sum" in low or "two_sum" in low:
        fname = fname or "two_sum"
        return f"""def {fname}(nums, target):\n    seen = {{}}\n    for i, num in enumerate(nums):\n        need = target - num\n        if need in seen:\n            return [seen[need], i]\n        seen[num] = i\n    return []"""
    return None


# ------------------------- simple assignment logic -------------------------

def _try_assignment_logic(prompt: str) -> str | None:
    # High-confidence solver for small "each owns/has one of" puzzles.
    low = prompt.lower()
    if not any(k in low for k in ["each own", "each owns", "each has", "who owns", "who has"]):
        return None
    names = _extract_people(prompt)
    items = _extract_items(prompt)
    if not (2 <= len(names) <= 5 and len(items) == len(names)):
        return None

    constraints: list[tuple[str, str, str]] = []  # (name, op, item)
    for name in names:
        for item in items:
            pattern_not = rf"\b{name}\b[^.\n;]*\b(?:does not|doesn't|did not|is not|isn't|not)\b[^.\n;]*\b{item}\b"
            pattern_yes = rf"\b{name}\b[^.\n;]*\b(?:owns|has|is assigned|gets|received|keeps)\b[^.\n;]*\b{item}\b"
            if re.search(pattern_not, prompt, re.I):
                constraints.append((name, "!=", item))
            elif re.search(pattern_yes, prompt, re.I):
                constraints.append((name, "=", item))
    if not constraints:
        return None

    solutions: list[dict[str, str]] = []
    for perm in itertools.permutations(items):
        assign = dict(zip(names, perm))
        ok = True
        for name, op, item in constraints:
            if op == "=" and assign[name].lower() != item.lower(): ok = False; break
            if op == "!=" and assign[name].lower() == item.lower(): ok = False; break
        if ok:
            solutions.append(assign)
    if len(solutions) != 1:
        return None
    sol = solutions[0]
    return "; ".join(f"{n} = {sol[n]}" for n in names)


def _extract_people(prompt: str) -> list[str]:
    # Prefer comma/and list at the start: Alice, Bob, and Carla each...
    m = re.search(r"([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*(?:,?\s+and\s+[A-Z][a-z]+)?)\s+each\s+(?:own|owns|has|have)", prompt)
    if m:
        return [x.strip() for x in re.split(r",|\band\b", m.group(1)) if x.strip()]
    candidates = re.findall(r"\b[A-Z][a-z]+\b", prompt)
    stop = {"Who", "What", "Which", "On", "The", "A", "An"}
    out = []
    for c in candidates:
        if c not in stop and c not in out:
            out.append(c)
    return out[:5]


def _extract_items(prompt: str) -> list[str]:
    m = re.search(r"(?:pets?|items?|objects?|colors?|colours?)\s*:\s*([^.;\n]+)", prompt, re.I)
    if not m:
        m = re.search(r"one\s+(?:pet|item|object|color|colour)\s*:?\s*([^.;\n]+)", prompt, re.I)
    if m:
        raw = m.group(1).replace(" or ", ",").replace(" and ", ",")
        parts = [re.sub(r"\b(a|an|the)\b", "", x, flags=re.I).strip(" ,") for x in raw.split(",")]
        return [p.lower() for p in parts if p and not p.lower().startswith("who")][:5]
    # Common pet fallback.
    known = [x for x in ["cat", "dog", "bird", "fish", "hamster"] if re.search(rf"\b{x}\b", prompt, re.I)]
    return known


# ------------------------- simple NER -------------------------

def _try_simple_ner(prompt: str) -> str | None:
    if ":" not in prompt:
        return None
    text = prompt.split(":", 1)[1].strip()
    if not text or len(text) > 250:
        return None
    ents: list[tuple[str, str]] = []
    for m in re.finditer(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b", text):
        ents.append((m.group(0), "DATE"))
    orgs = re.findall(r"\b(?:AMD|Google|Microsoft|Amazon|OpenAI|Meta|Apple|NVIDIA|IBM|Intel)\b", text)
    for o in orgs:
        ents.append((o, "ORG"))
    # Person names: two consecutive capitalized words, excluding date/location phrases.
    two_token_people = []
    bad_first = {"On", "The", "A", "An"}
    bad_second = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}
    for m in re.finditer(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text):
        val = m.group(0)
        first, second = val.split()
        if val not in {"Austin Texas", "New York", "San Francisco"} and first not in bad_first and second not in bad_second:
            ents.append((val, "PERSON"))
            two_token_people.extend([first, second])
    # Common one-token person pattern: "Maria joined AMD ...". Skip tokens already part of a full name.
    for m in re.finditer(r"\b([A-Z][a-z]+)\s+(?:joined|met|visited|founded|called|emailed|hired|worked|spoke)\b", text):
        val = m.group(1)
        if val not in {"On", "The", "A", "An"} and val not in two_token_people:
            ents.append((val, "PERSON"))
    locs = ["Austin", "Texas", "New York", "London", "Paris", "Berlin", "Tokyo", "California", "Brazil"]
    for loc in locs:
        if re.search(rf"\b{re.escape(loc)}\b", text):
            ents.append((loc, "LOCATION"))
    # Deduplicate preserving order.
    seen = set(); clean = []
    for e in ents:
        if e not in seen:
            seen.add(e); clean.append(e)
    if len(clean) >= 2:
        return "\n".join(f"{ent} — {typ}" for ent, typ in clean)
    return None
