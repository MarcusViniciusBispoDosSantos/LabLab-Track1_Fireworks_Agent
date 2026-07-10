from __future__ import annotations

import ast
import itertools
import operator
import re
from decimal import Decimal, ROUND_HALF_UP
from statistics import median
from .classifier import TaskType

POS = set('good great excellent amazing love loved fast clean helpful easy smooth perfect reliable happy satisfied impressive best wonderful fantastic clear intuitive useful enjoy enjoyed works like liked awesome'.split())
NEG = set('bad terrible awful hate hated slow broken buggy fails failed failure crash crashes error poor worst disappointed confusing unusable late expensive problem issue wrong difficult unreliable'.split())
NEGATIONS = set('not no never hardly barely without'.split())

def try_local(prompt: str, task_type: TaskType) -> str | None:
    try:
        if task_type == TaskType.SENTIMENT: return sentiment(prompt)
        if task_type == TaskType.MATH: return math(prompt)
        if task_type == TaskType.CODE_GEN: return code_gen(prompt)
        if task_type == TaskType.CODE_DEBUG: return code_debug(prompt)
        if task_type == TaskType.LOGIC: return assignment_logic(prompt)
        # NER local is intentionally disabled for broad hidden prompts; LLM is safer.
        return None
    except Exception:
        return None

# ---------------- sentiment ----------------
def sentiment(prompt: str) -> str | None:
    low = prompt.lower()
    if not any(k in low for k in ['sentiment', 'review', 'positive', 'negative', 'neutral', 'polarity', 'tone']): return None
    text = prompt.split(':', 1)[1] if ':' in prompt else prompt
    words = re.findall(r"[a-zA-Z']+", text.lower())
    pos = neg = 0
    for i, w in enumerate(words):
        if w in POS:
            if i and words[i-1] in NEGATIONS: neg += 1
            else: pos += 1
        if w in NEG:
            if i and words[i-1] in NEGATIONS: pos += 1
            else: neg += 1
    # contrast markers strongly suggest mixed if both clauses have affective content.
    if pos and neg or (pos and any(x in low for x in [' but ', ' however ', ' although ']) and any(n in words for n in NEG)):
        lab = 'Mixed'
    elif pos: lab = 'Positive'
    elif neg: lab = 'Negative'
    else: lab = 'Neutral'
    if 'justify' in low or 'explain' in low or 'why' in low:
        return f'{lab} — the text contains {lab.lower()} sentiment signals.'
    return lab

# ---------------- math ----------------
def math(prompt: str) -> str | None:
    if any(x in prompt for x in ['def ', 'return ', '```', 'function']): return None
    low = prompt.lower().replace(',', '')
    expr = extract_expr(prompt)
    if expr:
        try: return fmt(_safe_eval(expr))
        except Exception: pass
    nums = [Decimal(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", low)]
    if not nums: return None
    # percent of
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)", low)
    if m: return fmtd(Decimal(m.group(1)) * Decimal(m.group(2)) / 100)
    # increase/decrease X by Y percent
    m = re.search(r"(?:increase|raise|grow)\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\s*(?:%|percent)", low)
    if m: return fmtd(Decimal(m.group(1)) * (1 + Decimal(m.group(2))/100))
    m = re.search(r"(?:decrease|reduce|drop)\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\s*(?:%|percent)", low)
    if m: return fmtd(Decimal(m.group(1)) * (1 - Decimal(m.group(2))/100))
    # base grows by a% then loses b%
    if len(nums) >= 3 and any(k in low for k in ['grows', 'growth', 'increase', 'increases', 'rose']) and any(k in low for k in ['loses', 'loss', 'decrease', 'decreases', 'drops', 'fell', 'reduced']):
        return fmtd(nums[0] * (1 + nums[1]/100) * (1 - nums[2]/100))
    # monthly/yearly repeated growth: base, pct, periods
    if len(nums) >= 3 and any(k in low for k in ['each month', 'per month', 'monthly', 'each year', 'per year', 'annually']) and any(k in low for k in ['grow', 'increase', 'compound']):
        base, pct, periods = nums[0], nums[1]/100, int(nums[2])
        return fmtd(base * ((1 + pct) ** periods))
    # discount then tax
    if len(nums) >= 2 and any(k in low for k in ['discount', 'off']):
        val = nums[0] * (1 - nums[1]/100)
        if len(nums) >= 3 and 'tax' in low: val *= (1 + nums[2]/100)
        return fmtd(val)
    if any(k in low for k in ['average', 'mean']):
        return fmtd(sum(nums) / Decimal(len(nums))) if len(nums) >= 2 else None
    if 'median' in low and len(nums) >= 2:
        return fmt(float(median([float(x) for x in nums])))
    if any(k in low for k in ['sum', 'total of']) and len(nums) >= 2:
        return fmtd(sum(nums))
    return None

def extract_expr(prompt: str) -> str | None:
    candidates = re.findall(r"[0-9(][0-9\s+\-*/().%^]*[0-9)]", prompt)
    good = []
    for c in candidates:
        c = c.strip().replace('^', '**')
        if any(op in c for op in ['+', '-', '*', '/', '**', '%']) and re.fullmatch(r"[0-9\s+\-*/().*%]+", c): good.append(c)
    return max(good, key=len) if good else None

def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode='eval')
    return _eval(node.body)

def _eval(node):
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): return -_eval(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in ops: return ops[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError('bad expr')

def fmt(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 1e-10 else f'{x:.8f}'.rstrip('0').rstrip('.')

def fmtd(x: Decimal) -> str:
    q = x.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP).normalize()
    s = format(q, 'f')
    return s.rstrip('0').rstrip('.') if '.' in s else s

# ---------------- code ----------------
def fname(prompt: str, default: str) -> str:
    m = re.search(r"(?:function|method)\s+(?:called|named)\s+([A-Za-z_]\w*)", prompt, re.I)
    if m: return m.group(1)
    m = re.search(r"\b([A-Za-z_]\w*)\s*\([^)]*\)", prompt)
    return m.group(1) if m else default

def code_gen(prompt: str) -> str | None:
    low = prompt.lower()
    if not ('python' in low or 'function' in low or 'def ' in low): return None
    if 'palindrome' in low:
        f = fname(prompt, 'is_palindrome')
        return f"import re\n\ndef {f}(text):\n    cleaned = re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()\n    return cleaned == cleaned[::-1]"
    if 'factorial' in low:
        f = fname(prompt, 'factorial')
        return f"def {f}(n):\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"
    if 'fibonacci' in low:
        f = fname(prompt, 'fibonacci')
        return f"def {f}(n):\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"
    if 'prime' in low and ('true' in low or 'boolean' in low or 'is_prime' in low):
        f = fname(prompt, 'is_prime')
        return f"def {f}(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    i = 3\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 2\n    return True"
    if 'two sum' in low or 'two_sum' in low:
        f = fname(prompt, 'two_sum')
        return f"def {f}(nums, target):\n    seen = {{}}\n    for i, num in enumerate(nums):\n        need = target - num\n        if need in seen:\n            return [seen[need], i]\n        seen[num] = i\n    return []"
    return None

def code_debug(prompt: str) -> str | None:
    if 'len(num)' in prompt and re.search(r'def\s+\w+\s*\(\s*nums\s*\)', prompt):
        m = re.search(r"(def\s+\w+\s*\(\s*nums\s*\):[\s\S]+)", prompt)
        code = (m.group(1) if m else prompt).strip().replace('len(num)', 'len(nums)')
        return 'Bug: `num` is undefined; use `nums`.\n\n' + code
    if re.search(r'range\s*\(\s*len\((\w+)\)\s*\+\s*1\s*\)', prompt):
        fixed = re.sub(r'range\s*\(\s*len\((\w+)\)\s*\+\s*1\s*\)', r'range(len(\1))', prompt)
        m = re.search(r"(def\s+\w+\s*\([^)]*\):[\s\S]+)", fixed)
        return 'Bug: loop goes one past the last valid index.\n\n' + (m.group(1).strip() if m else fixed)
    return None

# ---------------- logic ----------------
def assignment_logic(prompt: str) -> str | None:
    low = prompt.lower()
    if not any(x in low for x in ['each owns', 'each own', 'each has', 'who owns', 'who has']): return None
    names = people(prompt); items = objects(prompt)
    if not (2 <= len(names) <= 5 and len(items) == len(names)): return None
    cons = []
    for n in names:
        for it in items:
            if re.search(rf"\b{re.escape(n)}\b[^.\n;]*(?:does not|doesn't|is not|isn't|not)[^.\n;]*\b{re.escape(it)}\b", prompt, re.I): cons.append((n, '!=', it))
            if re.search(rf"\b{re.escape(n)}\b[^.\n;]*(?:owns|has|gets|received|keeps)[^.\n;]*\b{re.escape(it)}\b", prompt, re.I): cons.append((n, '=', it))
    sols = []
    for perm in itertools.permutations(items):
        a = dict(zip(names, perm)); ok = True
        for n, op, it in cons:
            if op == '=' and a[n].lower() != it.lower(): ok = False
            if op == '!=' and a[n].lower() == it.lower(): ok = False
        if ok: sols.append(a)
    if len(sols) == 1:
        return '; '.join(f'{n} = {sols[0][n]}' for n in names)
    return None

def people(prompt: str) -> list[str]:
    m = re.search(r"([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*(?:,?\s+and\s+[A-Z][a-z]+)?)\s+each\s+(?:own|owns|has|have)", prompt)
    if m: return [x.strip() for x in re.split(r",|\band\b", m.group(1)) if x.strip()]
    out=[]
    for c in re.findall(r"\b[A-Z][a-z]+\b", prompt):
        if c not in {'Who','What','Which','The','A','An','On'} and c not in out: out.append(c)
    return out[:5]

def objects(prompt: str) -> list[str]:
    m = re.search(r"(?:pets?|items?|objects?|colors?|colours?)\s*:?\s*([^.;\n]+)", prompt, re.I)
    if not m: m = re.search(r"one\s+(?:pet|item|object|color|colour)\s*:?\s*([^.;\n]+)", prompt, re.I)
    if m:
        raw = m.group(1).replace(' or ', ',').replace(' and ', ',')
        parts = [re.sub(r"\b(a|an|the)\b", '', x, flags=re.I).strip(' ,') for x in raw.split(',')]
        return [p.lower() for p in parts if p and not p.lower().startswith('who')][:5]
    return [x for x in ['cat','dog','bird','fish','hamster'] if re.search(rf"\b{x}\b", prompt, re.I)]
