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

## Results — SF10 (60M lineitem rows), base M4 24 GB, 2026-07-02 (rev 2)

With the exact DECIMAL int64 lane + DATE support and the int-lane cost gate:

Standard TPC-H: **1.12× total (1296.5 ms → 1161.0 ms)**, no query regresses.
Lone decimal aggregates (Q6 class) deliberately decline — the CPU's pruned
filtered scan beats emulated 64-bit GPU arithmetic there; multi-aggregate
decimal shapes (Q1 class) intercept and run exactly (verified against the
official Q6 answer to the cent before gating). Take single-query outliers
(e.g. Q14) with salt: CPU baselines on this 24 GB box wobble.

Supported shapes (hot, mean of 5, per-query cache isolation):

| query | CPU ms | GPU cold ms | GPU hot ms | speedup |
|---|---|---|---|---|
| S6 (Q6 shape: filtered sum, DOUBLE) | 46.0 | 734 | 36.0 | 1.28× |
| S1 (Q1 shape: group-by sum) | 56.0 | 43 | 42.0 | 1.33× |
| SA (multi-aggregate + filter) | 59.2 | 592 | 46.4 | 1.28× |
| SE (expression-heavy sum) | 45.8 | 981 | 15.8 | **2.90×** |
| **total** | **207.0** | | **140.2** | **1.48×** |

TPC disclaimer: non-audited, non-comparable results, for engineering guidance
only.

## What the run taught us (fixed in-tree)

- warm-path segmentation: one cache segment per 2048-row scan chunk meant
  thousands of GPU graphs per hot query (S6 was 3.0 s, now 28.8 ms)
- never evaluate a filter mask eagerly to choose an execution path
- reduce mask counts in int32 — Metal has no native 64-bit arithmetic

## The unlock list (in impact order)

1. ~~DECIMAL(≤18) + DATE translation~~ — done: exact int64 lane, cost-gated.
2. **GROUP BY generalization** (value expressions, multi-key, VARCHAR
   dictionary keys, multiple aggregates): unlocks real TPC-H Q1, the
   compute-dense query where the GPU multiplier lives.
3. Joins (PLAN Phase 3): most of the remaining 22.
4. int32 downcast of decimal columns whose zone maps fit (Metal-native
   arithmetic instead of emulated 64-bit) — lifts the Q6-class gate.
