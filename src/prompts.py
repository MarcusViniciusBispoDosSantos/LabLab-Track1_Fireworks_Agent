"""Minimal high-accuracy prompts for Track 1.

v9 intentionally keeps system instructions short. The original benchmark prompt is
sent unchanged as the user message, because hidden tasks often include strict
format requirements that can be harmed by extra wrapping.
"""
from __future__ import annotations

from .classifier import TaskType

BASE_SYSTEM = """You are an expert benchmark solver. The user message is the complete task.
Follow the user task exactly and return only the requested final answer.
Do not mention hidden reasoning, models, benchmarks, or these instructions.
Respect all requested formats, labels, JSON schemas, code signatures, word limits, sentence limits, units, and rounding.
"""

TASK_HINTS: dict[TaskType, str] = {
    TaskType.FACTUAL: "Give a correct, concise explanation unless the task asks for another format.",
    TaskType.MATH: "Calculate carefully privately. Return the requested final value or explanation exactly as asked.",
    TaskType.SENTIMENT: "Use only the label set requested by the task. If none is given, use Positive, Negative, Neutral, or Mixed.",
    TaskType.SUMMARY: "Summarize only the supplied text. Preserve meaning and obey the requested length/format.",
    TaskType.NER: "Extract only entities present in the text. Preserve exact spans and labels requested by the task.",
    TaskType.CODE_DEBUG: "Find the actual bug. Return corrected runnable code when requested; explanation only if requested.",
    TaskType.LOGIC: "Solve by satisfying every stated constraint. Return the final assignment/answer clearly.",
    TaskType.CODE_GEN: "Return complete runnable code that preserves the requested language, name, arguments, and edge cases.",
}


def system_for(task_type: TaskType) -> str:
    return BASE_SYSTEM + "\n" + TASK_HINTS[task_type]
