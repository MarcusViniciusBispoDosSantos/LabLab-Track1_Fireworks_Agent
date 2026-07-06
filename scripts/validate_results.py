#!/usr/bin/env python3
"""Validate Track 1 /output/results.json format against /input/tasks.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="input/tasks.json")
    parser.add_argument("--results", default="output/results.json")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    results_path = Path(args.results)
    assert tasks_path.exists(), f"Missing tasks file: {tasks_path}"
    assert results_path.exists(), f"Missing results file: {results_path}"

    tasks = load_json(tasks_path)
    results = load_json(results_path)

    assert isinstance(tasks, list), "tasks.json must be a JSON array"
    assert isinstance(results, list), "results.json must be a JSON array"
    assert len(results) == len(tasks), f"Expected {len(tasks)} results, got {len(results)}"

    task_ids = {str(t.get("task_id")) for t in tasks if isinstance(t, dict)}
    result_ids = {str(r.get("task_id")) for r in results if isinstance(r, dict)}
    assert task_ids == result_ids, f"Task IDs mismatch. Missing={task_ids-result_ids}, Extra={result_ids-task_ids}"

    for r in results:
        assert isinstance(r, dict), f"Each result must be an object, got {type(r).__name__}"
        assert isinstance(r.get("task_id"), str), f"task_id must be string: {r}"
        assert isinstance(r.get("answer"), str), f"answer must be string: {r}"
        assert r["answer"].strip(), f"answer is empty for task_id={r.get('task_id')}"

    print("PASS: results.json is valid and all tasks have non-empty answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
