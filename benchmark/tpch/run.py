#!/usr/bin/env python3
"""TPC-H benchmark runner: CPU engine vs duckdb-mlx, GQE methodology.

- one persistent database file per scale factor (dbgen once)
- one duckdb session per mode so the GPU column cache persists across runs
  (it is process-lifetime); each query runs 1 cold + N hot repetitions
- standard TPC-H 22 queries measure fallback parity; the gpu_shapes suite
  measures acceleration on the shapes the extension supports today
"""
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DUCKDB = ROOT / "build/release/duckdb"
REPEATS = 5

GPU_SHAPES_SETUP = """
CREATE TABLE IF NOT EXISTS lineitem_dbl AS
SELECT l_linenumber,
       l_quantity::DOUBLE AS q,
       l_extendedprice::DOUBLE AS price,
       l_discount::DOUBLE AS disc,
       l_tax::DOUBLE AS tax,
       (l_shipdate - DATE '1970-01-01')::INTEGER AS shipdays
FROM lineitem;
"""

GPU_SHAPES = [
    ("S6 (Q6 shape: filtered sum)",
     "SELECT sum(price * disc) FROM lineitem_dbl "
     "WHERE shipdays >= 8766 AND shipdays < 9131 AND disc >= 0.049 AND disc <= 0.071 AND q < 24;"),
    ("S1 (Q1 shape: group-by sum)",
     "SELECT l_linenumber, sum(price * (1.0 - disc)) FROM lineitem_dbl "
     "GROUP BY l_linenumber ORDER BY l_linenumber;"),
    ("SA (multi-aggregate + filter)",
     "SELECT count(*), avg(price), min(price), max(price), sum(price * disc) "
     "FROM lineitem_dbl WHERE shipdays > 9131;"),
    ("SE (expression-heavy sum)",
     "SELECT sum(sqrt(abs(price)) + q * disc - tax / (q + 1.0)) FROM lineitem_dbl;"),
]


def official_queries(db):
    """The tpch extension's self-contained query texts (Q15 included)."""
    proc = subprocess.run([str(DUCKDB), str(db), "-json", "-c",
                           "LOAD tpch; SELECT query_nr, query FROM tpch_queries();"],
                          capture_output=True, text=True, check=True)
    import json
    rows = json.loads(proc.stdout)
    return [(f"Q{row['query_nr']}", row["query"]) for row in rows]


def run_mode(db, queries, gpu, setup=""):
    script = [".timer on", ".mode csv", f"SET mlx_execution = {'true' if gpu else 'false'};",
              "SET mlx_min_rows = 1000;"]
    if setup:
        script.append(setup)
    keys = {f"q{idx}": label for idx, (label, _) in enumerate(queries)}
    for idx, (label, sql) in enumerate(queries):
        if gpu:
            # isolate queries: each starts cold, and resident caches from
            # earlier queries don't build up memory pressure
            script.append("SELECT '##CLEAR##';")
            script.append("SELECT mlx_cache_clear();")
        for i in range(REPEATS + 1):
            script.append(f"SELECT '##q{idx}#{i}';")
            script.append(sql if sql.endswith(";") else sql + ";")
    proc = subprocess.run([str(DUCKDB), str(db)], input="\n".join(script),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"duckdb failed:\n{proc.stderr[-2000:]}")

    times = {}
    current = None
    pending_marker_time = False
    for line in proc.stdout.splitlines():
        if "##CLEAR##" in line:
            current = None
            continue
        marker = re.search(r"##(q\d+)#(\d+)", line)
        if marker:
            current = keys[marker.group(1)]
            pending_marker_time = True
            continue
        m = re.match(r"Run Time \(s\): real ([0-9.]+)", line)
        if m and current:
            if pending_marker_time:
                pending_marker_time = False  # the marker SELECT's own timing
            else:
                times.setdefault(current, []).append(float(m.group(1)) * 1000.0)
    return times


def report(title, queries, cpu, gpu):
    print(f"\n## {title}")
    print(f"| query | cpu hot ms | gpu cold ms | gpu hot ms | speedup |")
    print(f"|---|---|---|---|---|")
    total_cpu = total_gpu = 0.0
    for label, _ in queries:
        c = cpu.get(label, [])
        g = gpu.get(label, [])
        if not c or not g:
            print(f"| {label} | (missing) | | | |")
            continue
        cpu_hot = mean(c[1:]) if len(c) > 1 else c[0]
        gpu_cold = g[0]
        gpu_hot = mean(g[1:]) if len(g) > 1 else g[0]
        total_cpu += cpu_hot
        total_gpu += gpu_hot
        speed = cpu_hot / gpu_hot if gpu_hot > 0 else float("inf")
        print(f"| {label} | {cpu_hot:.1f} | {gpu_cold:.1f} | {gpu_hot:.1f} | {speed:.2f}x |")
    if total_gpu > 0:
        print(f"| **total** | **{total_cpu:.1f}** | | **{total_gpu:.1f}** | **{total_cpu / total_gpu:.2f}x** |")


def main():
    sf = sys.argv[1] if len(sys.argv) > 1 else "1"
    out = ROOT / "benchmark/tpch/results"
    out.mkdir(parents=True, exist_ok=True)
    db = out / f"tpch_sf{sf}.duckdb"
    if not db.exists():
        print(f"generating TPC-H SF{sf} ...", file=sys.stderr)
        subprocess.run([str(DUCKDB), str(db)], input=f"LOAD tpch; CALL dbgen(sf := {sf});",
                       capture_output=True, text=True, check=True)

    tpch_queries = official_queries(db)

    print(f"# TPC-H SF{sf} — DuckDB CPU vs duckdb-mlx (hot = mean of {REPEATS} runs)")

    cpu = run_mode(db, tpch_queries, gpu=False)
    gpu = run_mode(db, tpch_queries, gpu=True)
    report("Standard TPC-H (DECIMAL/DATE/VARCHAR: measures fallback parity + overhead)",
           tpch_queries, cpu, gpu)

    cpu_s = run_mode(db, GPU_SHAPES, gpu=False, setup=GPU_SHAPES_SETUP)
    gpu_s = run_mode(db, GPU_SHAPES, gpu=True, setup=GPU_SHAPES_SETUP)
    report("Supported shapes on TPC-H data (measures acceleration)", GPU_SHAPES, cpu_s, gpu_s)


if __name__ == "__main__":
    main()
