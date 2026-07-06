"""Heuristic task router for Track 1.

The judge uses unseen prompts, so this file does not hardcode answers. It only
classifies the broad capability category to choose a better prompt/model.
"""

from __future__ import annotations

import re
from enum import StrEnum


class TaskType(StrEnum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARY = "summary"
    NER = "ner"
    CODE_DEBUG = "code_debug"
    LOGIC = "logic"
    CODE_GEN = "code_generation"


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*%?")
_CODE_HINT_RE = re.compile(
    r"(```|\bdef\s+\w+\s*\(|\bclass\s+\w+|\bfunction\s+\w+|\bpublic\s+static\b|"
    r"\bTraceback\b|\bException\b|\bSyntaxError\b|\bTypeError\b|\bValueError\b|"
    r"\breturn\b|\bconsole\.log\b|\bSELECT\b|\bfor\s*\(|\bwhile\s*\()",
    re.IGNORECASE,
)


def classify_task(prompt: str) -> TaskType:
    p = prompt.strip()
    low = p.lower()

    if any(k in low for k in [
        "sentiment", "positive, negative", "negative, neutral", "classify the tone",
        "label the review", "is this review positive", "emotion expressed",
    ]):
        return TaskType.SENTIMENT

    if any(k in low for k in [
        "summarize", "summarise", "summary", "tl;dr", "tldr", "condense",
        "one-sentence", "one sentence summary", "bullet summary", "main idea",
    ]):
        return TaskType.SUMMARY

    if any(k in low for k in [
        "named entity", "named entities", "extract entities", "entity recognition",
        "label entities", "person, org", "person, organization", "locations and dates",
    ]):
        return TaskType.NER

    if any(k in low for k in [
        "debug", "bug", "fix the code", "correct the code", "what is wrong with this code",
        "why does this code", "traceback", "exception", "error in this code",
    ]):
        return TaskType.CODE_DEBUG

    # Code generation should be detected before math, because specs often include numbers.
    if _CODE_HINT_RE.search(p) and any(k in low for k in [
        "write", "implement", "create", "complete", "function", "method", "class",
        "program", "algorithm", "return", "code", "python", "javascript", "typescript",
        "java", "c++", "sql",
    ]):
        if not any(k in low for k in ["what is", "explain", "define"]):
            return TaskType.CODE_GEN

    if any(k in low for k in [
        "logic puzzle", "deduce", "deductive", "constraint", "constraints", "exactly one",
        "at least one", "at most one", "if and only if", "iff", "truth-teller", "liar",
        "arrange", "which person", "who owns", "which box", "must be true", "cannot be true",
    ]):
        return TaskType.LOGIC

    nums = _NUMBER_RE.findall(p)
    math_words = [
        "calculate", "compute", "solve", "what is", "how many", "how much", "percentage",
        "percent", "discount", "tax", "interest", "ratio", "average", "mean", "median",
        "probability", "increase", "decrease", "total", "remaining", "projection", "rate",
    ]
    if len(nums) >= 2 and any(w in low for w in math_words):
        return TaskType.MATH
    if re.search(r"\d+\s*[+\-*/^]\s*\d+", p) and not _CODE_HINT_RE.search(p):
        return TaskType.MATH

    if _CODE_HINT_RE.search(p) and any(k in low for k in ["bug", "fix", "correct", "output", "why"]):
        return TaskType.CODE_DEBUG

    return TaskType.FACTUAL
