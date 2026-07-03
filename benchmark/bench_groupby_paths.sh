#!/usr/bin/env bash
# Small-scale GROUP BY path benchmarks — run locally before M3 Ultra scale-up.
# Usage: ./benchmark/bench_groupby_paths.sh [duckdb_binary]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DUCKDB="${1:-$ROOT/build/release/duckdb}"
EXT="$ROOT/build/release/extension/mlx/mlx.duckdb_extension"

if [[ ! -x "$DUCKDB" ]]; then
  echo "duckdb not found: $DUCKDB" >&2
  exit 1
fi

run_bench() {
  local label="$1"
  local path="$2"
  local rows="$3"
  local groups="$4"
  "$DUCKDB" -unsigned <<SQL
LOAD '$EXT';
.timer on
CREATE OR REPLACE TABLE t AS
SELECT (range % $groups)::BIGINT AS g, (range * 3)::BIGINT AS v
FROM range($rows);
SET mlx_groupby_path = '$path';
SELECT '$label' AS bench, mlx_groupby_bench(list(g), list(v)) AS checksum FROM t;
SELECT '$label' AS bench, mlx_groupby_bench(list(g), list(v)) AS checksum_hot FROM t;
SQL
}

echo "=== mlx GROUP BY path microbench (small scale) ==="
run_bench "dense_20k_100"   dense     20000   100
run_bench "slotlock_100k_2k" slotlock 100000  2000
run_bench "radix_80k_50k"   radix    80000  50000
run_bench "sort_60k_12k"    sort     60000  12000
run_bench "auto_150k_3k"    auto    150000  3000
echo "=== done ==="
