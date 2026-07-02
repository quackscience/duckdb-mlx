#!/usr/bin/env python3
"""TPC-H benchmark runner aligned with NVIDIA GQE methodology.

GQE reference: https://github.com/rapidsai/gqe
  - load all TPC-H tables once (tables resident in GPU memory)
  - run all 22 standard TPC-H queries (q1.sql … q22.sql)
  - average each query over 5 hot-cache repetitions
  - report per-query and total execution time vs DuckDB CPU

Our DuckDB/MLX mapping:
  - one persistent .duckdb file per scale factor (dbgen once)
  - one duckdb process per mode (CPU / GPU) so session state persists
  - GPU warm-up: one untimed pass of all 22 queries to populate the column
    cache (analogous to GQE load_tpch.py + first touch)
  - timed phase: 5 hot repetitions per query, no mlx_cache_clear between queries

Usage:
  python3 benchmark/tpch/run.py 10           # GQE-style SF10 (default mode)
  python3 benchmark/tpch/run.py 10 --isolated  # dev: per-query cache isolation
  python3 benchmark/tpch/run.py 10 --shapes-only # supported-shape micro-suite
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DUCKDB = ROOT / "build/release/duckdb"
GQE_REPEATS = 5

# GQE load_tpch.py equivalent: resident GPU columns before timed queries.
TPCH_PIN = "SELECT mlx_cache_pin_tpch();"

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


def official_queries(db: Path) -> list[tuple[str, str]]:
    """Standard TPC-H 22 queries via DuckDB's tpch extension."""
    proc = subprocess.run(
        [str(DUCKDB), str(db), "-json", "-c",
         "LOAD tpch; SELECT query_nr, query FROM tpch_queries() ORDER BY query_nr;"],
        capture_output=True, text=True, check=True,
    )
    rows = json.loads(proc.stdout)
    return [(f"Q{row['query_nr']}", row["query"].strip()) for row in rows]


def parse_timings(stdout: str, keys: dict[str, str]) -> dict[str, list[float]]:
    times: dict[str, list[float]] = {}
    current: str | None = None
    pending_marker = False
    for line in stdout.splitlines():
        if "##CLEAR##" in line:
            current = None
            continue
        marker = re.search(r"##(q\d+)#(\d+)", line)
        if marker:
            current = keys[marker.group(1)]
            pending_marker = True
            continue
        m = re.match(r"Run Time \(s\): real ([0-9.]+)", line)
        if m and current:
            if pending_marker:
                pending_marker = False
            else:
                times.setdefault(current, []).append(float(m.group(1)) * 1000.0)
    return times


def build_script(
    queries: list[tuple[str, str]],
    gpu: bool,
    *,
    repeats: int,
    isolated: bool,
    warmup: bool,
    setup: str = "",
    pin_tpch: bool = False,
) -> tuple[list[str], dict[str, str]]:
    script = [
        "LOAD duckdb_mlx;",
        "LOAD tpch;",
        ".timer on",
        ".mode csv",
        f"SET mlx_execution = {'true' if gpu else 'false'};",
        "SET mlx_min_rows = 1000;",
    ]
    if setup:
        script.append(setup)
    if gpu and pin_tpch and not isolated:
        script.append(TPCH_PIN)

    keys = {f"q{idx}": label for idx, (label, _) in enumerate(queries)}

    if warmup:
        script.append("SELECT '##WARMUP##';")
        for _, sql in queries:
            script.append(sql if sql.endswith(";") else sql + ";")

    for idx, (_, sql) in enumerate(queries):
        if gpu and isolated:
            script.append("SELECT '##CLEAR##';")
            script.append("SELECT mlx_cache_clear();")
        for i in range(repeats):
            script.append(f"SELECT '##q{idx}#{i}';")
            script.append(sql if sql.endswith(";") else sql + ";")
    return script, keys


def run_session(
    db: Path,
    queries: list[tuple[str, str]],
    gpu: bool,
    *,
    repeats: int,
    isolated: bool,
    warmup: bool,
    setup: str = "",
    pin_tpch: bool = False,
) -> dict[str, list[float]]:
    script, keys = build_script(
        queries, gpu, repeats=repeats, isolated=isolated, warmup=warmup, setup=setup, pin_tpch=pin_tpch,
    )
    proc = subprocess.run(
        [str(DUCKDB), str(db)], input="\n".join(script), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"duckdb failed:\n{proc.stderr[-4000:]}")
    return parse_timings(proc.stdout, keys)


def report_gqe(title: str, queries: list[tuple[str, str]], cpu: dict[str, list[float]],
               gpu: dict[str, list[float]]) -> None:
    """GQE-style table: per-query hot means + total wall time."""
    print(f"\n## {title}")
    print("| query | cpu hot ms | gpu hot ms | speedup |")
    print("|---|---|---|---|")
    total_cpu = total_gpu = 0.0
    for label, _ in queries:
        c = cpu.get(label, [])
        g = gpu.get(label, [])
        if not c or not g:
            print(f"| {label} | (missing) | (missing) | |")
            continue
        cpu_hot = mean(c)
        gpu_hot = mean(g)
        total_cpu += cpu_hot
        total_gpu += gpu_hot
        speed = cpu_hot / gpu_hot if gpu_hot > 0 else float("inf")
        print(f"| {label} | {cpu_hot:.1f} | {gpu_hot:.1f} | {speed:.2f}x |")
    if total_gpu > 0:
        print(f"| **total (22 queries)** | **{total_cpu:.1f}** | **{total_gpu:.1f}** | "
              f"**{total_cpu / total_gpu:.2f}x** |")


def report_dev(title: str, queries: list[tuple[str, str]], cpu: dict[str, list[float]],
               gpu: dict[str, list[float]]) -> None:
    """Dev table with cold + hot columns (isolated-cache mode)."""
    print(f"\n## {title}")
    print("| query | cpu hot ms | gpu cold ms | gpu hot ms | speedup |")
    print("|---|---|---|---|---|")
    total_cpu = total_gpu = 0.0
    for label, _ in queries:
        c = cpu.get(label, [])
        g = gpu.get(label, [])
        if not c or not g:
            print(f"| {label} | (missing) | | | |")
            continue
        cpu_hot = mean(c[1:]) if len(c) > 1 else mean(c)
        gpu_cold = g[0]
        gpu_hot = mean(g[1:]) if len(g) > 1 else g[0]
        total_cpu += cpu_hot
        total_gpu += gpu_hot
        speed = cpu_hot / gpu_hot if gpu_hot > 0 else float("inf")
        print(f"| {label} | {cpu_hot:.1f} | {gpu_cold:.1f} | {gpu_hot:.1f} | {speed:.2f}x |")
    if total_gpu > 0:
        print(f"| **total** | **{total_cpu:.1f}** | | **{total_gpu:.1f}** | "
              f"**{total_cpu / total_gpu:.2f}x** |")


def ensure_db(sf: str) -> Path:
    out = ROOT / "benchmark/tpch/results"
    out.mkdir(parents=True, exist_ok=True)
    db = out / f"tpch_sf{sf}.duckdb"
    if not db.exists():
        print(f"generating TPC-H SF{sf} ...", file=sys.stderr)
        subprocess.run(
            [str(DUCKDB), str(db)],
            input=f"LOAD tpch; CALL dbgen(sf := {sf});",
            capture_output=True, text=True, check=True,
        )
    return db


def main() -> None:
    parser = argparse.ArgumentParser(description="TPC-H benchmark (GQE-aligned)")
    parser.add_argument("sf", nargs="?", default="1", help="scale factor (default 1)")
    parser.add_argument("--repeats", type=int, default=GQE_REPEATS,
                        help=f"timed repetitions per query (GQE uses {GQE_REPEATS})")
    parser.add_argument("--isolated", action="store_true",
                        help="clear GPU cache before each query (dev mode, not GQE-fair)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip GPU/CPU warm-up pass (not GQE-fair)")
    parser.add_argument("--shapes-only", action="store_true",
                        help="run only the supported-shape micro-suite")
    args = parser.parse_args()

    if not DUCKDB.is_file():
        sys.exit(f"build/release/duckdb not found — run `make release` first")

    db = ensure_db(args.sf)
    warmup = not args.no_warmup
    repeats = args.repeats + (1 if args.isolated else 0)  # isolated: 1 cold + N hot

    if args.shapes_only:
        queries = GPU_SHAPES
        setup = GPU_SHAPES_SETUP
        # Mixed projections on lineitem_dbl need per-query cache isolation until
        # the column cache handles varying scan projections on the same table.
        if not args.isolated:
            args.isolated = True
            print("# shapes-only: using --isolated (mixed projection cache)", file=sys.stderr)
    else:
        queries = official_queries(db)
        setup = ""

    mode = "isolated (per-query cache clear)" if args.isolated else "GQE-aligned (resident cache)"
    print(f"# TPC-H SF{args.sf} — DuckDB CPU vs duckdb-mlx")
    print(f"# methodology: {mode}")
    print(f"# queries: {len(queries)} | timed repeats/query: {args.repeats} | warm-up: {warmup}")
    print("# reference: https://github.com/rapidsai/gqe (22 queries, hot-cache averages, total wall time)")
    print()

    cpu = run_session(db, queries, gpu=False, repeats=repeats, isolated=False,
                      warmup=warmup, setup=setup, pin_tpch=False)
    gpu = run_session(db, queries, gpu=True, repeats=repeats, isolated=args.isolated,
                      warmup=warmup, setup=setup, pin_tpch=not args.shapes_only and not args.isolated)

    if args.isolated:
        report_dev("Standard TPC-H" if not args.shapes_only else "Supported shapes", queries, cpu, gpu)
    else:
        report_gqe("Standard TPC-H (22 queries, GQE methodology)" if not args.shapes_only
                   else "Supported shapes (GQE resident-cache methodology)", queries, cpu, gpu)


if __name__ == "__main__":
    main()
