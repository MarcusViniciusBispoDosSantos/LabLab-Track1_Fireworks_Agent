"""Heuristic task router for AMD Hackathon Track 1.

The router does not hardcode answers. It detects the broad capability category so
FireRoute AI can use the best prompt and model for unseen tasks.
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
_CODE_BLOCK_RE = re.compile(
    r"```|\bdef\s+\w+\s*\(|\bclass\s+\w+|\bfunction\s+\w+|\breturn\b|\bimport\s+\w+|"
    r"\bconsole\.log\b|\bSELECT\b|\bfor\s*\(|\bwhile\s*\(|\{\s*\n|=>|;\s*$",
    re.I | re.M,
)
_ERROR_RE = re.compile(
    r"\b(debug|bug|fix|error|exception|traceback|syntaxerror|typeerror|valueerror|indexerror|"
    r"wrong output|fails?|failure|broken|incorrect|correct(?:ed)? implementation|does not work)\b",
    re.I,
)
_CODE_GEN_RE = re.compile(
    r"\b(write|implement|create|complete|generate|provide|build|design)\b.{0,120}"
    r"\b(function|method|class|program|script|algorithm|query|sql|python|javascript|typescript|java|c\+\+|code)\b|"
    r"\b(function|method|class)\s+(called|named|that|which)\b|"
    r"\breturn\s+(true|false|a|an|the|list|array|string|integer|number|dict|object)\b",
    re.I | re.S,
)


def classify_task(prompt: str) -> TaskType:
    p = prompt.strip()
    low = p.lower()

    # Explicit sentiment instructions first; they often contain negative words like
    # "fails" or "broken" that should not trigger code debugging.
    if any(k in low for k in [
        "sentiment", "positive, negative", "negative, neutral", "positive/negative",
        "positive or negative", "classify the tone", "label the review", "review as",
        "is this review positive", "emotion expressed", "polarity", "mood of the text",
    ]):
        return TaskType.SENTIMENT

    # Summary detection must require an explicit summarization intent.
    # Do NOT route prompts like "Explain HTTP in two sentences" to SUMMARY;
    # those are factual tasks with a length constraint.
    if any(k in low for k in [
        "summarize", "summarise", "summary", "tl;dr", "tldr", "condense", "recap",
        "bullet summary", "key points", "main idea", "shorten the following",
        "compress the following", "summarise the following", "summarize the following",
        "summarise this", "summarize this", "summarise the text", "summarize the text",
    ]):
        return TaskType.SUMMARY

    if any(k in low for k in [
        "named entity", "named entities", "extract entities", "entity recognition", "label entities",
        "extract all entities", "identify all entities", "list all entities", "person, org",
        "person, organization", "person, organisation", "people, organizations", "locations and dates",
        "names, dates", "dates, locations", "ner", "entities from", "entities in",
    ]):
        return TaskType.NER
    if any(t in low for t in ["person", "organization", "organisation", "location", "date"]):
        if any(v in low for v in ["extract", "identify", "label", "list", "tag"]):
            return TaskType.NER

    # Debugging: explicit bug/fix instruction or error words with code context.
    explicit_debug = any(k in low for k in [
        "debug", "find and fix", "find the bug", "what is wrong with this code",
        "why does this code", "correct this code", "identify the bug", "fix the following code",
        "debug the following", "runtime error", "syntaxerror", "typeerror", "valueerror",
        "indexerror", "traceback", "wrong output", "does not work", "throws an error",
    ])
    code_context = bool(_CODE_BLOCK_RE.search(p)) or any(k in low for k in [
        "this function", "this method", "this class", "python function", "javascript function",
        "typescript function", "java method", "code snippet", "implementation", "program", "script",
    ])
    if explicit_debug or (_ERROR_RE.search(p) and code_context):
        return TaskType.CODE_DEBUG

    if _CODE_GEN_RE.search(p) or any(k in low for k in [
        "write a python function", "write a function", "implement a function", "create a function",
        "write code", "generate code", "complete the function", "write a sql query",
        "create a class", "implement the algorithm", "write a program", "provide code",
    ]):
        return TaskType.CODE_GEN

    logic_keywords = [
        "logic puzzle", "deduce", "deductive", "constraint", "constraints", "exactly one",
        "at least one", "at most one", "if and only if", " iff ", "truth-teller", "liar",
        "arrange", "which person", "who owns", "who has", "which box", "must be true",
        "cannot be true", "each own", "each owns", "each has", "sits next to", "is older than",
        "is taller than", "left of", "right of", "next to", "different", "not the same",
    ]
    if any(k in low for k in logic_keywords):
        return TaskType.LOGIC

    nums = _NUMBER_RE.findall(p)
    math_words = [
        "calculate", "compute", "solve", "evaluate", "how many", "how much", "percentage",
        "percent", "discount", "tax", "interest", "ratio", "average", "mean", "median",
        "probability", "increase", "decrease", "total", "remaining", "projection", "rate",
        "grows", "growth", "loss", "after", "before", "per", "cost", "price", "revenue",
        "profit", "distance", "speed", "time", "hours", "minutes", "fraction", "sum",
        "product", "difference", "twice", "triple", "half", "quarter", "more than", "less than",
    ]
    if len(nums) >= 2 and any(w in low for w in math_words):
        return TaskType.MATH
    if re.search(r"\d+\s*[+\-*/^]\s*\d+", p) and not _CODE_BLOCK_RE.search(p):
        return TaskType.MATH

    return TaskType.FACTUAL
