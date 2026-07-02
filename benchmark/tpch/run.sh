#!/usr/bin/env bash
# TPC-H benchmark runner — GQE methodology (5 hot runs, same-machine CPU vs GPU).
set -euo pipefail

SF="${1:-1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${BUILD:-$ROOT/build/release}"
DUCKDB="${DUCKDB:-$BUILD/duckdb}"
EXT="${EXT:-$BUILD/extension/duckdb_mlx.duckdb_extension}"
OUT="$ROOT/benchmark/tpch/results/sf${SF}"
mkdir -p "$OUT"

if [[ ! -x "$DUCKDB" ]]; then
  echo "duckdb binary not found at $DUCKDB — run: GEN=ninja make release" >&2
  exit 1
fi

echo "=== TPC-H SF${SF} benchmark ==="
echo "duckdb: $DUCKDB"
echo "extension: $EXT"
echo "output: $OUT"

# Generate dataset
"$DUCKDB" -unsigned <<SQL
LOAD '$EXT';
LOAD tpch;
CALL dbgen(sf := ${SF});
SQL

# Extract individual queries (split on -- QN markers)
QUERIES_FILE="$ROOT/benchmark/tpch/queries.sql"
mapfile -t QUERY_BLOCKS < <(awk '
  /^-- Q[0-9]+$/ { if (q) print q; q=$0; next }
  { q = q "\n" $0 }
  END { if (q) print q }
' "$QUERIES_FILE")

CPU_MS=()
GPU_MS=()
LABELS=()

run_timed() {
  local mode="$1"
  local sql="$2"
  local ms
  ms=$("$DUCKDB" -unsigned -c "LOAD '$EXT'; SET mlx_execution = ${mode}; SET mlx_min_rows = 0; .timer on" -c "$sql" 2>&1 \
    | awk '/^Run Time/ {print $(NF-1); exit}')
  echo "${ms:-0}"
}

for block in "${QUERY_BLOCKS[@]}"; do
  label=$(echo "$block" | head -1 | tr -d '# ')
  sql=$(echo "$block" | tail -n +2)
  LABELS+=("$label")

  echo "--- ${label} (CPU baseline) ---"
  total_cpu=0
  for i in 1 2 3 4 5; do
    t=$(run_timed false "$sql")
    total_cpu=$(echo "$total_cpu + $t" | bc -l)
  done
  avg_cpu=$(echo "scale=3; $total_cpu / 5" | bc -l)
  CPU_MS+=("$avg_cpu")

  echo "--- ${label} (GPU) ---"
  # cold run (ingest)
  cold=$(run_timed true "$sql")
  total_gpu=0
  for i in 1 2 3 4 5; do
    t=$(run_timed true "$sql")
    total_gpu=$(echo "$total_gpu + $t" | bc -l)
  done
  avg_gpu=$(echo "scale=3; $total_gpu / 5" | bc -l)
  GPU_MS+=("$avg_gpu")

  speedup="n/a"
  if [[ "$avg_gpu" != "0" && "$avg_gpu" != "0.000" ]]; then
    speedup=$(echo "scale=2; $avg_cpu / $avg_gpu" | bc -l)
  fi

  {
    echo "${label} cpu_ms=${avg_cpu} gpu_cold_ms=${cold} gpu_hot_ms=${avg_gpu} speedup=${speedup}"
  } >> "$OUT/per_query.txt"
done

# Aggregate report
python3 "$ROOT/benchmark/tpch/report.py" "$OUT/per_query.txt" > "$OUT/report.txt"
cat "$OUT/report.txt"
