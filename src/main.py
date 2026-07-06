from __future__ import annotations

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .agent import AgentConfig, FireworksTrack1Agent


def _load_dotenv_if_present(path: Path = Path(".env")) -> None:
    """Small .env loader for local development only.

    The submitted Docker image should not contain .env. This loader simply makes local
    testing convenient without adding third-party dependencies.
    """
    if os.getenv("DISABLE_DOTENV", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("/input/tasks.json must contain a JSON array")
    tasks: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            tasks.append(item)
    return tasks


def _write_results(path: Path, results: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def main() -> int:
    # Local development only. In the submitted image there should be no bundled .env file.
    _load_dotenv_if_present()

    input_path = Path(os.getenv("INPUT_PATH", "/input/tasks.json"))
    output_path = Path(os.getenv("OUTPUT_PATH", "/output/results.json"))

    try:
        tasks = _load_tasks(input_path)
        config = AgentConfig.from_env()
        agent = FireworksTrack1Agent(config)

        max_workers = int(os.getenv("MAX_WORKERS", "4"))
        max_workers = max(1, min(max_workers, 8, len(tasks) or 1))

        indexed_results: list[tuple[int, dict[str, str]]] = []
        if max_workers == 1 or len(tasks) <= 1:
            for idx, task in enumerate(tasks):
                indexed_results.append((idx, agent.answer_task(task)))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(agent.answer_task, task): idx for idx, task in enumerate(tasks)}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        indexed_results.append((idx, future.result()))
                    except Exception as exc:
                        # Per-task fallback keeps the whole submission from producing malformed/missing JSON.
                        task_id = str(tasks[idx].get("task_id", f"task_{idx}"))
                        indexed_results.append((idx, {"task_id": task_id, "answer": f"Unable to complete task: {exc}"}))

        indexed_results.sort(key=lambda x: x[0])
        _write_results(output_path, [r for _, r in indexed_results])
        return 0
    except Exception:
        # A global setup/input/output failure should be a real failure, per the guide.
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
