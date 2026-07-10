from __future__ import annotations

import re
from enum import StrEnum

class TaskType(StrEnum):
    FACTUAL = 'factual'
    MATH = 'math'
    SENTIMENT = 'sentiment'
    SUMMARY = 'summary'
    NER = 'ner'
    CODE_DEBUG = 'code_debug'
    LOGIC = 'logic'
    CODE_GEN = 'code_generation'

_CODE_RE = re.compile(r"```|\bdef\s+\w+\s*\(|\bfunction\s+\w+|\bclass\s+\w+|\breturn\b|\bimport\s+\w+|=>|console\.log|;\s*$", re.I | re.M)
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*%?")

def classify_task(prompt: str) -> TaskType:
    p = prompt.strip()
    low = p.lower()

    # Sentiment first; hidden prompts often contain negative words that look like debugging.
    if any(x in low for x in [
        'sentiment', 'classify the review', 'classify this review', 'positive, negative',
        'positive or negative', 'negative or positive', 'neutral or', 'mixed sentiment',
        'polarity', 'tone of the text', 'is the following review', 'label the sentiment'
    ]):
        return TaskType.SENTIMENT

    if any(x in low for x in ['summarize', 'summarise', 'summary', 'tl;dr', 'tldr', 'condense', 'recap', 'key points', 'main idea', 'shorten the following']):
        return TaskType.SUMMARY

    if any(x in low for x in ['named entity', 'named entities', 'extract entities', 'entity recognition', 'label entities', 'person, org', 'person, organization', 'person, organisation', 'ner']):
        return TaskType.NER
    if any(x in low for x in ['extract', 'identify', 'list', 'label', 'tag']) and any(y in low for y in ['person', 'organization', 'organisation', 'location', 'date', 'entities']):
        return TaskType.NER

    explicit_debug = any(x in low for x in ['debug', 'find and fix', 'find the bug', 'fix the bug', 'what is wrong with this code', 'correct this code', 'throws an error', 'traceback', 'syntaxerror', 'typeerror', 'valueerror', 'indexerror', 'wrong output'])
    if explicit_debug or (any(x in low for x in ['bug', 'broken', 'fails', 'error', 'incorrect']) and _CODE_RE.search(p)):
        return TaskType.CODE_DEBUG

    if re.search(r"\b(write|implement|create|complete|generate|provide|build)\b.{0,120}\b(function|method|class|program|script|algorithm|query|code|python|javascript|typescript|java|sql)\b", low, re.S):
        return TaskType.CODE_GEN
    if any(x in low for x in ['write a function', 'write a python function', 'implement a function', 'complete the function', 'create a class', 'write code', 'write a sql query']):
        return TaskType.CODE_GEN

    if any(x in low for x in ['logic puzzle', 'deduce', 'deductive', 'constraint', 'exactly one', 'truth-teller', 'liar', 'who owns', 'who has', 'must be true', 'cannot be true', 'each owns', 'each has', 'left of', 'right of', 'next to', 'older than', 'taller than']):
        return TaskType.LOGIC

    nums = _NUM_RE.findall(p)
    if len(nums) >= 2 and any(x in low for x in ['calculate', 'compute', 'solve', 'evaluate', 'how many', 'how much', 'percent', 'percentage', 'discount', 'tax', 'interest', 'ratio', 'average', 'mean', 'median', 'probability', 'increase', 'decrease', 'total', 'remaining', 'projection', 'rate', 'growth', 'grows', 'loss', 'after', 'per', 'cost', 'price', 'revenue', 'profit', 'distance', 'speed', 'time', 'sum', 'product', 'difference', 'twice', 'triple']):
        return TaskType.MATH
    if re.search(r"\d+\s*[+\-*/^]\s*\d+", p) and not _CODE_RE.search(p):
        return TaskType.MATH

    return TaskType.FACTUAL
