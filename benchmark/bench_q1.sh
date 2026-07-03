#!/usr/bin/env bash
# TPC-H Q1 hot-cache benchmark (CPU vs GPU, pinned lineitem).
# Usage: benchmark/bench_q1.sh [sf] [repeats]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build/release}"
DUCKDB="${DUCKDB:-$BUILD/duckdb}"
SF="${1:-1}"
REPEATS="${2:-3}"
DB="$ROOT/benchmark/tpch/results/tpch_sf${SF}.duckdb"

if [[ ! -x "$DUCKDB" ]]; then
  echo "Build first: GEN=ninja make release" >&2
  exit 1
fi
if [[ ! -f "$DB" ]]; then
  echo "Missing $DB" >&2
  exit 1
fi

export ROOT DUCKDB DB REPEATS
python3 <<'PY'
import json, os, re, statistics, subprocess

root = os.environ["ROOT"]
duckdb = os.environ["DUCKDB"]
db = os.environ["DB"]
repeats = int(os.environ["REPEATS"])

proc = subprocess.run(
    [duckdb, db, "-json", "-c", "LOAD tpch; SELECT query FROM tpch_queries() WHERE query_nr=1;"],
    capture_output=True, text=True, check=True,
)
q1 = json.loads(proc.stdout)[0]["query"].strip()
if not q1.endswith(";"):
    q1 += ";"

script = [
    "LOAD mlx;", "LOAD tpch;", ".timer on",
    "SET mlx_execution = true;", "SET mlx_min_rows = 1000;",
    "SELECT mlx_cache_pin_tpch();",
    q1,  # warmup
]
for mode in ("cpu", "gpu"):
    script.append(f"SET mlx_execution = {'true' if mode == 'gpu' else 'false'};")
    for i in range(repeats):
        script.append(f"SELECT '##{mode}{i}##';")
        script.append(q1)

proc = subprocess.run([duckdb, db], input="\n".join(script), capture_output=True, text=True)
if proc.returncode:
    print(proc.stderr[-4000:])
    raise SystemExit(proc.returncode)

times = {"cpu": [], "gpu": []}
cur = None
pinned = False
for line in proc.stdout.splitlines():
    if "Run Time (s): real" in line:
        t = float(line.split("real")[1].split()[0]) * 1000
        if not pinned and t > 1000:
            pinned = True
            cur = None
            continue
        if pinned and cur and t >= 1.0:
            times[cur].append(t)
            cur = None
        continue
    m = re.search(r"##(cpu|gpu)\d+##", line)
    if m and pinned:
        cur = m.group(1)

sf = os.path.basename(db).replace("tpch_sf", "").replace(".duckdb", "")
print(f"TPC-H Q1 SF{sf} ({repeats} hot repeats each mode)")
print(f"  CPU hot avg: {statistics.mean(times['cpu']):.1f} ms  ({', '.join(f'{x:.1f}' for x in times['cpu'])})")
print(f"  GPU hot avg: {statistics.mean(times['gpu']):.1f} ms  ({', '.join(f'{x:.1f}' for x in times['gpu'])})")
sp = statistics.mean(times["cpu"]) / statistics.mean(times["gpu"])
print(f"  Speedup (CPU/GPU hot): {sp:.2f}x")
PY
