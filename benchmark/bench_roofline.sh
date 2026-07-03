#!/usr/bin/env bash
# Roofline microbench on pinned resident columns (SUM + multi-agg fusion).
# Usage: benchmark/bench_roofline.sh [sf] [col_suffix]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build/release}"
DUCKDB="${DUCKDB:-$BUILD/duckdb}"
SF="${1:-1}"
COL="${2:-lineitem#5}"
DB="$ROOT/benchmark/tpch/results/tpch_sf${SF}.duckdb"
REPEATS="${REPEATS:-3}"

if [[ ! -x "$DUCKDB" ]]; then
  echo "Build first: GEN=ninja make release" >&2
  exit 1
fi
if [[ ! -f "$DB" ]]; then
  echo "Missing $DB" >&2
  exit 1
fi

export DUCKDB DB COL REPEATS SF
python3 <<'PY'
import os, re, subprocess, statistics

duckdb = os.environ["DUCKDB"]
db = os.environ["DB"]
col = os.environ["COL"]
repeats = int(os.environ["REPEATS"])
sf = os.environ["SF"]

script = [
    "LOAD mlx;", "SET mlx_execution = true;",
    "SELECT mlx_cache_pin('lineitem');",
]
for fn in ("mlx_stream_sum_bench", "mlx_multi_agg_bench"):
    for i in range(repeats):
        script.append(f"SELECT '##{fn}#{i}##';")
        script.append(f"SELECT {fn}('{col}');")

proc = subprocess.run([duckdb, db], input="\n".join(script), capture_output=True, text=True)
if proc.returncode:
    print(proc.stderr[-3000:])
    raise SystemExit(proc.returncode)

results = {fn: [] for fn in ("mlx_stream_sum_bench", "mlx_multi_agg_bench")}
cur = None
for line in proc.stdout.splitlines():
    m = re.search(r"##(mlx_stream_sum_bench|mlx_multi_agg_bench)#\d+##", line)
    if m:
        cur = m.group(1)
        continue
    if cur and "gib_s=" in line:
        results[cur].append(line.strip())
        cur = None

print(f"Roofline bench SF{sf} col={col}")
for fn in results:
    print(f"\n## {fn}")
    for line in results[fn]:
        print(f"  {line}")

def gib(line):
    m = re.search(r"gib_s=([0-9.]+)", line)
    return float(m.group(1)) if m else None

print()
for fn, lines in results.items():
    vals = [gib(l) for l in lines if gib(l)]
    if not vals:
        print(f"{fn}: (no data)")
        continue
    hot = statistics.mean(vals[1:]) if len(vals) > 1 else vals[0]
    print(f"{fn}: cold={vals[0]:.1f} GiB/s  hot_avg={hot:.1f} GiB/s")
PY
