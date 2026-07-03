# TPC-H benchmark (GQE-aligned)

## Quick repeatable suite

```shell
GEN=ninja make release
benchmark/run_all.sh 1          # minimal + roofline + Q1 @ SF1
benchmark/bench_roofline.sh 1   # stream SUM + multi-agg fusion (GiB/s)
benchmark/bench_q1.sh 1         # TPC-H Q1 CPU vs GPU (pinned)
benchmark/bench_q1.sh 10        # SF10 (pin ~2 min)
python3 benchmark/tpch/run.py 10  # full 22-query GQE harness
```

```shell
python3 benchmark/tpch/run.py 10     # pins all 8 TPC-H tables, then 22 queries
python3 benchmark/tpch/run.py 1      # SF1 (fast sanity check)
```

Inside duckdb, GQE-style load:

```sql
LOAD mlx;
SELECT mlx_cache_pin_tpch();          -- all 8 tables
-- or: SELECT mlx_cache_pin('lineitem');
-- returns [rows, columns, already_resident]
```

## Methodology (matches [NVIDIA GQE](https://github.com/rapidsai/gqe))

This harness mirrors the TPC-H workflow described in the GQE README and the
[NVIDIA GQE blog post](https://developer.nvidia.com/blog/designing-gpu-accelerated-query-engines-with-nvidia-gqe/):

| GQE | duckdb-mlx equivalent |
|---|---|
| `load_tpch.py` — COPY all 8 tables into GPU memory once | `SELECT mlx_cache_pin_tpch();` (or per-table `mlx_cache_pin('lineitem')`) |
| Tables resident until node manager exits | One duckdb process per mode; GPU column cache persists |
| `run_tpch.py … all` — 22 standard queries | `tpch_queries()` — official DuckDB TPC-H texts |
| 5 hot-cache repetitions per query (blog) | `--repeats 5` (default), after one warm-up pass |
| Total wall time over all 22 queries | **total** row in output table |
| DuckDB CPU baseline | `SET mlx_execution = false` |

**Not GQE-fair (dev only):** `--isolated` clears `mlx_cache_clear()` before each
query. Use this to measure cold intercept paths, not to compare against GQE headline numbers.

**Supported-shape micro-suite:** `--shapes-only` runs the DOUBLE/INTEGER shapes
the extension accelerates today (S6, S1, SA, SE).

## Options

```shell
python3 benchmark/tpch/run.py 10 --repeats 5          # default
python3 benchmark/tpch/run.py 10 --no-warmup        # skip warm-up (not GQE-fair)
python3 benchmark/tpch/run.py 10 --isolated         # per-query cache clear (dev)
python3 benchmark/tpch/run.py 10 --shapes-only      # micro-suite only
```

## Results — SF10 (60M lineitem rows), base M4 24 GB, 2026-07-02 (rev 3, GQE-aligned)

**Methodology:** one warm-up pass of all 22 queries, then 5 timed hot repetitions
per query, no `mlx_cache_clear()` between queries (matches GQE resident-table
semantics). Reference: [rapidsai/gqe](https://github.com/rapidsai/gqe).

| | CPU total | GPU total | speedup |
|---|---|---|---|
| **22 queries (SF10)** | **3434.8 ms** | **7071.8 ms** | **0.49×** |

GPU wins on 17/22 queries individually (typical 1.2–1.9×), but **Q1 dominates**
the total: 196 ms CPU vs **4559 ms GPU** (filtered GROUP BY on DECIMAL, cold
scan every repetition because `table_filters` skip full-table cache pin). That
single query accounts for the aggregate loss. Q6 with resident cache: 52 ms CPU
vs 30 ms GPU (1.76×).

Supported shapes (`--shapes-only`, isolated cache, DOUBLE/INTEGER casts):

| query | CPU ms | GPU ms | speedup |
|---|---|---|---|
| S6 (Q6 shape: filtered sum) | 46.0 | 36.0 | 1.28× |
| S1 (Q1 shape: group-by sum) | 56.0 | 42.0 | 1.33× |
| SA (multi-aggregate + filter) | 59.2 | 46.4 | 1.28× |
| SE (expression-heavy sum) | 45.8 | 15.8 | **2.90×** |
| **total** | **207.0** | **140.2** | **1.48×** |

Rev 2 (pre-GQE-alignment, per-query cache clear) showed 1.12× on standard TPC-H
total — that mixed cold/hot semantics and is not comparable to GQE headline numbers.

TPC disclaimer: non-audited, non-comparable results, for engineering guidance
only. GQE headline numbers (SF1000, GB200) use different hardware and full GPU
table residency via Parquet COPY; compare methodology, not absolute milliseconds.

## What the run taught us (fixed in-tree)

- warm-path segmentation: one cache segment per 2048-row scan chunk meant
  thousands of GPU graphs per hot query (S6 was 3.0 s, now 28.8 ms)
- never evaluate a filter mask eagerly to choose an execution path
- reduce mask counts in int32 — Metal has no native 64-bit scatter_add
- GQE-fair runs must **not** clear the GPU cache between queries

## The unlock list (in impact order)

1. ~~DECIMAL(≤18) + DATE translation~~ — done: exact int64 lane, cost-gated.
2. **GROUP BY generalization** (value expressions, multi-key, VARCHAR
   dictionary keys, multiple aggregates): unlocks real TPC-H Q1, the
   compute-dense query where the GPU multiplier lives.
3. Joins (PLAN Phase 3): most of the remaining 22.
4. int32 downcast of decimal columns whose zone maps fit (Metal-native
   arithmetic instead of emulated 64-bit) — lifts the Q6-class gate.
