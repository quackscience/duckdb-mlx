# duckdb-mlx

GPU-accelerated DuckDB on Apple Silicon (Metal / MLX).

`duckdb_mlx` is a loadable DuckDB extension that executes analytical workloads on the
Apple Silicon GPU. See [docs/PLAN.md](docs/PLAN.md) for the full architecture and
roadmap; the design ports concepts from NVIDIA GQE and Sirius, re-derived for unified
memory. Progress is test-gated: every capability lands with a passing differential test
against DuckDB's own CPU results before it is committed.

**Status:** Transparent GPU acceleration is live. An optimizer hook intercepts supported
plans, and a GPU-resident column cache (the GQE "in-memory table format", unified-memory
edition) serves repeated queries with **no table scan at all**, fused into a few Metal
kernels by `mx::compile`.

## Headline: plain SQL, 14× (base M4, 100M rows, hot cache)

```sql
LOAD 'duckdb_mlx';
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;  -- that's it. no special syntax.
```

| Query (100M rows) | DuckDB CPU (all cores) | duckdb-mlx GPU (hot) | Speedup |
|---|---|---|---|
| `sum(sin(x)*cos(x)+sqrt(abs(x)+1))` | 190 ms | **13 ms** | **14×** |
| `sum(x)` | 13 ms | **4 ms** | 3.3× (fp32 bandwidth limit) |

First run over a table is cold (~0.5 s: scan + GPU cache build), then the columns stay
GPU-resident; row-count changes invalidate and repopulate automatically. During hot GPU
queries the CPU does ~1 ms of work versus 1.6 CPU-seconds on the CPU engine. Every result
is verified against DuckDB's own engine by the differential test suite. Reproduce with
`benchmark/bench_transparent.sql`.

## Flagship: GPU-resident vector search

```sql
LOAD 'duckdb_mlx';

-- one-time: pin an embedding column as a GPU-resident, L2-normalized matrix
SELECT mlx_vss_pin('items', list(emb ORDER BY id)) FROM items;

-- cosine top-k, one matvec against the pinned matrix
SELECT idx, score FROM mlx_vss_search('items', [0.12, 0.34, ...], 10);

-- Q queries in a single matmul (reranking / batch scoring / similarity joins)
SELECT query_no, idx, score FROM mlx_vss_search_batch('items', [[...], [...]], 10);
```

Measured on a base M4 (24 GB), 1M vectors × 384 dims fp32 (~1.5 GB), hot cache,
verified result-identical to DuckDB's `array_cosine_similarity` brute force:

| Workload | DuckDB CPU (all cores) | duckdb-mlx GPU | Speedup |
|---|---|---|---|
| 1 query, top-10 | 70 ms | 20 ms | 3.5× (at bandwidth floor) |
| 32 queries | ~1.7 s | ~105 ms | **16×** |
| 128 queries | ~6.5 s | ~375 ms | **17×** |

Pin cost is ~5 s one-time and amortizes across all queries. Reproduce with
`benchmark/bench_vss.sql` and `benchmark/bench_vss_batch.sql`.

## All functions

| Function | Purpose |
|---|---|
| `mlx_info()` | extension/GPU/MLX/spdlog status string |
| `mlx_selftest()` | end-to-end GPU sanity check, returns `ok` |
| `mlx_vss_pin(name, list(col))` | pin a `FLOAT[N]` column as a GPU-resident matrix |
| `mlx_vss_search(name, query, k)` | cosine top-k against a pinned matrix |
| `mlx_vss_search_batch(name, queries, k)` | batched top-k, one matmul |
| `mlx_sum(BIGINT[])`, `mlx_expr_bench(BIGINT[])` | Phase 0 spike/benchmark vehicles |

Settings: `mlx_execution` (reserved for the Phase 1 optimizer hook), `mlx_min_rows`,
`mlx_log_level`.

The extension builds on all DuckDB platforms; the GPU path is compiled in only on Apple
Silicon (`DUCKDB_MLX_ENABLE_GPU`, auto-detected) and GPU functions raise clean errors
elsewhere.

## Repository layout

```
src/
├── duckdb_mlx_extension.cpp   # load/register, settings, spike functions
├── ops/mlx_vss.cpp            # vector-search scalar + table functions
├── bridge/mlx_bridge.cpp      # the only TU that includes MLX headers
├── mlx_logger.cpp             # the only TU that includes spdlog headers
├── planner/ format/ kernels/ exec/   # reserved for Phase 1+ (see docs/PLAN.md §5)
└── include/                   # bridge/logger interfaces (no MLX/spdlog types)
test/sql/                      # sqllogictests incl. GPU-vs-CPU differentials
benchmark/                     # reproduction scripts for the numbers above
third_party/mlx/               # MLX v0.31.2 submodule (Apple-only, not vcpkg)
```

Isolation rule (learned the hard way): DuckDB's vendored fmt shadows the real fmt for
every extension target, so MLX and spdlog each live in a dedicated static library whose
inherited include paths are cleared — no translation unit ever includes both `duckdb.hpp`
and MLX/spdlog headers. Data crosses the bridge as raw unified-memory pointers.

## Building

Portable C++ dependencies are managed with [vcpkg](https://vcpkg.io) (`vcpkg.json`;
currently spdlog). MLX is vendored as a submodule and built automatically into
`build/mlx-install` on first configure (`scripts/build_mlx.sh`). Requires macOS 14+,
Xcode (for the Metal compiler), CMake and ninja.

```shell
git clone https://github.com/Microsoft/vcpkg.git && ./vcpkg/bootstrap-vcpkg.sh
export VCPKG_TOOLCHAIN_PATH=`pwd`/vcpkg/scripts/buildsystems/vcpkg.cmake

git submodule update --init          # duckdb v1.5.4, extension-ci-tools, mlx v0.31.2
GEN=ninja make release
```

Main artifacts:

```shell
./build/release/duckdb                                            # shell, extension pre-loaded
./build/release/test/unittest                                     # test runner
./build/release/extension/duckdb_mlx/duckdb_mlx.duckdb_extension  # loadable extension
```

## Testing

```shell
GEN=ninja make test
```

SQL logic tests live in `test/sql/`; every GPU capability is diffed against DuckDB's CPU
results (the permanent differential harness from docs/PLAN.md Phase 1).

## License

MIT
