#!/usr/bin/env python3
"""Summarize TPC-H per-query benchmark results (GQE-style aggregate speedup)."""
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = dict(p.split("=", 1) for p in line.split())
        q = parts.get("Q1", parts.get("Q2", ""))  # fallback
        label = line.split()[0]
        cpu = float(parts["cpu_ms"])
        gpu = float(parts["gpu_hot_ms"])
        speedup = cpu / gpu if gpu > 0 else 0.0
        rows.append((label, cpu, gpu, speedup))

    total_cpu = sum(r[1] for r in rows)
    total_gpu = sum(r[2] for r in rows)
    agg = total_cpu / total_gpu if total_gpu > 0 else 0.0
    ge3 = sum(1 for r in rows if r[3] >= 3.0)

    print("TPC-H benchmark summary (5-run hot average, same-machine DuckDB vs duckdb-mlx)")
    print(f"queries={len(rows)} aggregate_speedup={agg:.2f}x queries_ge_3x={ge3}")
    print()
    print(f"{'query':<6} {'cpu_ms':>10} {'gpu_ms':>10} {'speedup':>8}")
    print("-" * 38)
    for label, cpu, gpu, speedup in rows:
        print(f"{label:<6} {cpu:10.3f} {gpu:10.3f} {speedup:7.2f}x")


if __name__ == "__main__":
    main()
