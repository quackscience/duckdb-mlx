#!/usr/bin/env bash
# GROUP BY CPU vs GPU at scales where slot-lock/radix should matter.
# Usage: ./benchmark/bench_groupby_scale.sh [duckdb_binary]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DUCKDB="${1:-$ROOT/build/release/duckdb}"
EXT="$ROOT/build/release/extension/mlx/mlx.duckdb_extension"

if [[ ! -x "$DUCKDB" ]]; then
  echo "duckdb not found: $DUCKDB" >&2
  exit 1
fi

run_case() {
  local label="$1"
  local rows="$2"
  local groups="$3"
  "$DUCKDB" -unsigned <<SQL
LOAD '$EXT';
.timer on
CREATE OR REPLACE TABLE t AS
SELECT (range % $groups)::BIGINT AS g, (range * 3)::BIGINT AS v
FROM range($rows);

SELECT '--- $label cpu ---' AS section;
SET mlx_execution = false;
SELECT sum(s) FROM (SELECT g, sum(v) AS s FROM t GROUP BY g) q;
SELECT sum(s) FROM (SELECT g, sum(v) AS s FROM t GROUP BY g) q;

SELECT '--- $label gpu auto ---' AS section;
SET mlx_execution = true;
SET mlx_groupby_path = 'auto';
SELECT sum(s) FROM (SELECT g, sum(v) AS s FROM t GROUP BY g) q;
SELECT sum(s) FROM (SELECT g, sum(v) AS s FROM t GROUP BY g) q;
SELECT sum(s) FROM (SELECT g, sum(v) AS s FROM t GROUP BY g) q;

SELECT '--- $label gpu slotlock ---' AS section;
SET mlx_groupby_path = 'slotlock';
SELECT mlx_groupby_bench(list(g), list(v)) FROM t;
SELECT mlx_groupby_bench(list(g), list(v)) FROM t;
SELECT mlx_groupby_bench(list(g), list(v)) FROM t;
SQL
}

echo "=== mlx GROUP BY scale bench ($(uname -m)) ==="
echo "Compare CPU transparent GROUP BY vs GPU auto (cached after run 1) vs forced slotlock bench."
echo ""

run_case "500k_2k"   500000   2000
run_case "1M_10k"  1000000  10000
run_case "5M_50k"  5000000  50000

echo "=== done ==="
