from __future__ import annotations

from .classifier import TaskType

# Tiny system prompts keep proxy token usage low. The original task is sent unchanged.
BASE = 'Answer the task accurately. Follow the requested format exactly. Return final answer only. No hidden reasoning.'
HINTS = {
    TaskType.FACTUAL: ' Be concise and factually correct.',
    TaskType.MATH: ' Compute carefully. Include steps only if asked. Put the final value clearly.',
    TaskType.SENTIMENT: ' Use only the requested sentiment label set; otherwise use Positive, Negative, Neutral, or Mixed.',
    TaskType.SUMMARY: ' Summarize only the provided text and obey length constraints.',
    TaskType.NER: ' Extract exact entity spans and labels requested.',
    TaskType.CODE_DEBUG: ' Identify the bug and provide corrected code when appropriate.',
    TaskType.LOGIC: ' Satisfy every constraint exactly.',
    TaskType.CODE_GEN: ' Provide complete runnable code with the requested name/signature.',
}

def system_for(task_type: TaskType) -> str:
    return BASE + HINTS.get(task_type, '')
