# TPC-H benchmark

```shell
python3 benchmark/tpch/run.py 1     # scale factor (1, 10, ...)
```

One database file per scale factor (dbgen once, reused); one duckdb session per
mode so the process-lifetime GPU cache can be hot; each query runs 1 cold + 5
hot repetitions (GQE methodology). Two suites:

1. **Standard TPC-H 22 queries** (official `tpch_queries()` texts). TPC-H
   columns are DECIMAL/DATE/VARCHAR, which the translator does not support yet,
   so this suite measures **fallback parity and interception overhead**.
2. **Supported shapes** on `lineitem` casted to DOUBLE/INTEGER
   (`lineitem_dbl`), measuring **acceleration** on the shapes the extension
   handles today.

## Results — SF10 (60M lineitem rows), base M4 24 GB, 2026-07-02

Standard TPC-H: **1.02× total (1061.6 ms → 1038.3 ms)** — every query declines
cleanly, results correct, no measurable optimizer-hook overhead.

Supported shapes (hot, mean of 5):

| query | CPU ms | GPU cold ms | GPU hot ms | speedup |
|---|---|---|---|---|
| S6 (Q6 shape: filtered sum) | 36.8 | 674 | 28.8 | 1.28× |
| S1 (Q1 shape: group-by sum(expr)) | 42.2 | 43 | 41.2 | 1.02× (declines: group-by value expressions unsupported) |
| SA (multi-aggregate + filter) | 52.0 | 55 | 43.6 | 1.19× |
| SE (expression-heavy sum) | 44.8 | 1045 | 15.8 | **2.84×** |
| **total** | **175.8** | | **129.4** | **1.36×** |

TPC disclaimer: non-audited, non-comparable results, for engineering guidance
only.

## What the run taught us (fixed in-tree)

- warm-path segmentation: one cache segment per 2048-row scan chunk meant
  thousands of GPU graphs per hot query (S6 was 3.0 s, now 28.8 ms)
- never evaluate a filter mask eagerly to choose an execution path
- reduce mask counts in int32 — Metal has no native 64-bit arithmetic

## The unlock list (in impact order)

1. **DECIMAL(≤18) + DATE translation** (scaled int64 / epoch days): standard
   TPC-H queries become interceptable at all.
2. **GROUP BY value expressions** (`sum(price * (1 - disc))`) and multi-key /
   VARCHAR-dictionary group keys: Q1-shape acceleration.
3. Joins (PLAN Phase 3): most of the remaining 22.
