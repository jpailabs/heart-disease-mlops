#!/usr/bin/env bash
# Hit every endpoint of a running API. Usage: scripts/smoke_test.sh [base_url] [n_requests]
set -euo pipefail
BASE=${1:-http://localhost:8088}
N=${2:-50}
cd "$(dirname "$0")/.."
echo "GET  /health      -> $(curl -sf "$BASE/health")"
echo "GET  /ready       -> $(curl -sf "$BASE/ready")"
echo "POST /predict     -> $(curl -sf -X POST "$BASE/predict" -H 'content-type: application/json' -d @sample_request.json)"
echo ">> sending $N predictions drawn from the cleaned dataset"
python3 - "$BASE" "$N" <<'PY'
import csv, json, random, sys, urllib.request
base, n = sys.argv[1], int(sys.argv[2])
rows = list(csv.DictReader(open("data/processed/heart_clean.csv")))
random.seed(7)
ok = 0
for row in random.choices(rows, k=n):
    rec = {k: (None if v == "" else float(v)) for k, v in row.items() if k != "target"}
    for k in ("age", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"):
        if rec[k] is not None:
            rec[k] = int(rec[k])
    req = urllib.request.Request(base + "/predict", json.dumps(rec).encode(),
                                 {"content-type": "application/json"})
    ok += urllib.request.urlopen(req).status == 200
print(f"{ok}/{n} predictions OK")
PY
# one deliberately invalid payload to show validation + 4xx metrics
curl -s -o /dev/null -w "invalid payload    -> HTTP %{http_code}\n" -X POST "$BASE/predict" \
     -H 'content-type: application/json' -d '{"age": -1}'
