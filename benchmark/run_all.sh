#!/usr/bin/env bash
# Run the full repeatable benchmark suite (minimal + roofline + Q1).
# Usage: benchmark/run_all.sh [sf]
#   sf: TPC-H scale factor for tpch-based benches (default 1)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SF="${1:-1}"

echo "========================================"
echo " mlx benchmark suite  SF=${SF}"
echo " $(uname -m)  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================"

echo ""
echo ">>> 1/3 minimal workloads"
"$ROOT/benchmark/run_minimal.sh"

echo ""
echo ">>> 2/3 roofline (stream sum + multi-agg)"
"$ROOT/benchmark/bench_roofline.sh" "$SF"

echo ""
echo ">>> 3/3 TPC-H Q1"
"$ROOT/benchmark/bench_q1.sh" "$SF"

echo ""
echo "Done."
