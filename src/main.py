from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any
from .agent import AgentConfig, FireworksTrack1Agent

def load_dotenv(path: Path = Path('.env')) -> None:
    if os.getenv('DISABLE_DOTENV','0').lower() in {'1','true','yes'} or not path.exists(): return
    for line in path.read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k,v=line.split('=',1); k=k.strip(); v=v.strip().strip('"').strip("'")
        if k and k not in os.environ: os.environ[k]=v

def load_tasks(path: Path) -> list[dict[str, Any]]:
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list): raise ValueError('/input/tasks.json must be a JSON array')
    return [x for x in data if isinstance(x, dict)]

def write_results(path: Path, results: list[dict[str,str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)

def main() -> int:
    load_dotenv()
    try:
        tasks=load_tasks(Path(os.getenv('INPUT_PATH','/input/tasks.json')))
        agent=FireworksTrack1Agent(AgentConfig.from_env())
        results=agent.answer_tasks(tasks)
        # Preserve input order and ensure every result has strings.
        clean=[]
        for i, task in enumerate(tasks):
            r=results[i] if i < len(results) else {}
            clean.append({'task_id': str(r.get('task_id') or task.get('task_id') or f'task_{i}'), 'answer': str(r.get('answer','')).strip()})
        write_results(Path(os.getenv('OUTPUT_PATH','/output/results.json')), clean)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
