import json, sys
from pathlib import Path
inp=Path(sys.argv[1] if len(sys.argv)>1 else 'input/tasks.json')
out=Path(sys.argv[2] if len(sys.argv)>2 else 'output/results.json')
tasks=json.loads(inp.read_text())
results=json.loads(out.read_text())
assert isinstance(tasks,list), 'tasks must be list'
assert isinstance(results,list), 'results must be list'
assert len(results)==len(tasks), f'expected {len(tasks)} results, got {len(results)}'
tids={str(t.get('task_id')) for t in tasks}
rids={str(r.get('task_id')) for r in results}
assert tids==rids, f'task_id mismatch missing={tids-rids} extra={rids-tids}'
for r in results:
    assert isinstance(r.get('answer'), str) and r.get('answer').strip(), f'bad answer: {r}'
print('PASS: valid /output/results.json')
