#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
python train.py --config configs/smoke.json
python evaluate.py --config configs/smoke.json --checkpoint checkpoints/smoke_model.pt
