#!/usr/bin/env bash
# Run the same stages as .github/workflows/ci.yml locally (lint -> test -> train -> docker).
# Fails fast on the first error, like the real pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

step "lint (flake8)"
$PY -m flake8 src api tests scripts --count --statistics
step "unit tests (pytest)"
$PY -m pytest -q --cov=src/heart --cov=api --cov-report=term
step "data + train (isolated MLflow store in a temp dir)"
$PY scripts/download_data.py
WORK=$(mktemp -d)
ROOT=$(pwd)
(cd "$WORK" && "$ROOT/$PY" -m heart.train --tracking-uri "sqlite:///$WORK/ci.db" \
    --experiment ci-local --model-dir "$WORK/models" 2>&1 | grep -E "heart.train") 
step "quality gate"
$PY -c "import json,sys;m=json.load(open('$WORK/models/model_metadata.json'))['metrics'];print('test_roc_auc =',m['test_roc_auc']);sys.exit(0 if m['test_roc_auc']>=0.85 else 1)"
step "docker build + smoke test"
docker build -q -t heart-api:ci .
docker rm -f heart-api-ci >/dev/null 2>&1 || true
docker run -d --name heart-api-ci -p 8099:8000 heart-api:ci >/dev/null
for i in $(seq 1 30); do curl -sf localhost:8099/ready >/dev/null && break; sleep 1; done
curl -sf -X POST localhost:8099/predict -H 'content-type: application/json' -d @sample_request.json; echo
docker rm -f heart-api-ci >/dev/null
step "pipeline passed"
