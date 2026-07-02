# TPC-H benchmark harness (GQE methodology)

Reproduces the measurement approach from NVIDIA GQE's TPC-H blog post:

- Same-machine DuckDB CPU vs duckdb-mlx GPU
- 5 hot-cache runs averaged (cold ingest reported separately)
- Per-query speedup and aggregate totals

## Prerequisites

```shell
GEN=ninja make release
```

Build with the `tpch` extension loaded (see `extension_config.cmake`).

## Quick run (SF1)

```shell
./benchmark/tpch/run.sh 1
```

## Output

`benchmark/tpch/results/sf<N>/report.txt` — per-query CPU ms, GPU ms, speedup, aggregate.

## Supported queries today

Transparent GPU execution currently covers **ungrouped aggregates** on single-table
scans with numeric filters. The runner marks unsupported queries as `SKIP` and still
records CPU baseline times for the full 22-query suite.

As GROUP BY, joins, and ORDER BY land, queries flip from SKIP to GPU automatically.
