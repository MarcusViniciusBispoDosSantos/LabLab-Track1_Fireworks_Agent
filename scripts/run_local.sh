#!/usr/bin/env bash
set -euo pipefail
export INPUT_PATH="${INPUT_PATH:-./samples/tasks_all_categories.json}"
export OUTPUT_PATH="${OUTPUT_PATH:-./samples/results.json}"
python -m src.main
python scripts/validate_results.py --tasks "$INPUT_PATH" --results "$OUTPUT_PATH"
cat "$OUTPUT_PATH"
