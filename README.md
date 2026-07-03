# duckdb-mlx

GPU-accelerated DuckDB on Apple Silicon (Metal / MLX).

The loadable extension is named **`mlx`** (`LOAD mlx;`). See [docs/PLAN.md](docs/PLAN.md) for
architecture and roadmap. Progress is test-gated: every capability lands with a passing
differential test against DuckDB's CPU engine before it is committed.

**Status (2026-07-03):** Transparent GPU acceleration is live. Optimizer intercepts supported
plans; GPU-resident column cache serves repeated queries without rescanning. TPC-H Q1 uses a
**pin-time bundle + fused Metal grid-stride kernel** (fast path; see benchmarks below). Roofline
microbenches (`mlx_stream_sum_bench`, `mlx_multi_agg_bench`) reach **~50 GiB/s SF1 / ~400+ GiB/s
SF10** on pinned lineitem — grouped Q1 still **~15–20 ms SF1** (~1.2× CPU); dedicated low-card
kernel is next (see PLAN §2a).

## Quick benchmark (repeatable)

```shell
GEN=ninja make release
./build/release/test/unittest "[sql]"          # 211 assertions, 12 SQL files

benchmark/run_all.sh 1                         # minimal + roofline + Q1 @ SF1
benchmark/bench_roofline.sh 1                  # stream SUM + multi-agg GiB/s
benchmark/bench_q1.sh 1                        # official TPC-H Q1 CPU vs GPU (pinned)
python3 benchmark/tpch/run.py 1                # full 22-query GQE harness
```

**SF10:** use `bench_roofline.sh 10` and `bench_q1.sh 10` with **lineitem-only pin** inside
those scripts. Full `mlx_cache_pin_tpch()` on SF10 can hit Metal memory limits on 24 GB machines.

Debug Q1 fast path: `SET mlx_log_level=debug;` — look for `MLX Q1 fast path: fused grid-stride`.

Reference Metal patterns: [gpudb / duckdbgpumetaldbram](https://github.com/singhpratech/duckdbgpumetaldbram).

## Headline: plain SQL, 14× (base M4, 100M rows, hot cache)

```sql
LOAD mlx;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;  -- plain SQL, no special syntax
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

**Transparent coverage so far:** ungrouped `sum` / `count` / `count(*)` / `avg` / `min` /
`max` over expressions of `+ - * /`, `sin`, `cos`, `sqrt`, `abs`, casts and constants on
DOUBLE/FLOAT/BIGINT/INTEGER columns — including **WHERE clauses**, which execute as GPU
masks over the resident columns (both pushed-down comparisons and residual predicates
with `AND`/`OR`/`NOT`). NULL semantics are honored throughout; anything unsupported
declines silently to DuckDB's CPU engine, and correctness always wins over speed
(the GPU pipeline computes in fp32 — counts are exact, floating aggregates agree with
the CPU engine to ~1e-7 relative in testing).

## Flagship: GPU-resident vector search

```sql
LOAD mlx;

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
| `mlx_cache_pin(table)` / `mlx_cache_pin_tpch()` | GPU-resident column cache |
| `mlx_stream_sum_bench(col_key)` | roofline SUM on pinned column (`lineitem#5`) |
| `mlx_multi_agg_bench(col_key)` | roofline SUM+MIN+MAX+COUNT fusion on pinned column |
| `mlx_vss_pin(name, list(col))` | pin a `FLOAT[N]` column as a GPU-resident matrix |
| `mlx_vss_search(name, query, k)` | cosine top-k against a pinned matrix |
| `mlx_vss_search_batch(name, queries, k)` | batched top-k, one matmul |
| `mlx_sum(BIGINT[])`, `mlx_expr_bench(BIGINT[])` | Phase 0 spike/benchmark vehicles |

Settings: `mlx_execution` (opt-in GPU plans), `mlx_min_rows`, `mlx_log_level`.

The extension builds on all DuckDB platforms; the GPU path is compiled in only on Apple
Silicon (`DUCKDB_MLX_ENABLE_GPU`, auto-detected). CI distribution builds **osx_arm64 only**.

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

Requires **macOS 14+**, **full Xcode.app** (Apple's IDE — not Command Line Tools alone),
CMake, and ninja. MLX compiles Metal shaders during the build.

**There is no Homebrew formula for the Metal compiler.** `brew install cmake ninja`
is fine; **Xcode itself must come from Apple** (App Store or developer download).

Check the toolchain before `make`:

```shell
./scripts/check_toolchain.sh
```

You need `xcrun -sdk macosx metal --version` to succeed.

### Install Xcode (first time — no Xcode on disk yet)

Use **one** of these. Do **not** use `brew install xcodesorg/made/xcodes` for the
first install — that Homebrew formula fails without Xcode already present (`xcbuild`
missing).

1. **App Store** — search “Xcode”, Install, open once to finish setup.

2. **Terminal via `mas`** (optional):

```shell
brew install mas
mas install 497799835
open -a Xcode
```

3. **[Apple Developer downloads](https://developer.apple.com/download/all/)** — download
   the `.xip`, extract, drag `Xcode.app` to `/Applications`, open once.

Then activate the toolchain:

```shell
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept
xcrun -sdk macosx metal --version
```

If Xcode is named `Xcode-16.x.app`, run `./scripts/check_toolchain.sh` for the exact
`xcode-select` path.

The [`xcodes`](https://github.com/XcodesOrg/xcodes) CLI is only useful **after** you
already have one Xcode — for installing/switching versions side by side.

Portable C++ dependencies (currently **spdlog**) are declared in `vcpkg.json` and
installed automatically on first build — `make` bootstraps a local `vcpkg/` checkout
if needed. MLX is vendored as a submodule and built into `build/mlx-install` on first
configure (`scripts/build_mlx.sh`, Metal-only — no JACCL/CPU backend).

```shell
git submodule update --init --recursive
GEN=ninja make release
```

If an earlier `make release` failed before vcpkg was set up, clear the stale CMake
cache and rebuild:

```shell
rm -rf build/release
GEN=ninja make release
```

If the vendored MLX build failed (e.g. after an Xcode/SDK upgrade), clear the MLX
artifacts and retry:

```shell
rm -rf build/mlx-build build/mlx-install
GEN=ninja make release
```

(`make release` also detects and removes poisoned caches automatically.)

To use a system-wide vcpkg install instead, set `VCPKG_TOOLCHAIN_PATH` before
`make` (see [DuckDB extension vcpkg docs](duckdb/extension/README.md)).

Main artifacts:

```shell
./build/release/duckdb                                            # shell, extension pre-loaded
./build/release/test/unittest                                     # test runner
./build/release/extension/mlx/mlx.duckdb_extension  # loadable extension
```

## Testing

```shell
GEN=ninja make test
# or:
./build/release/test/unittest "[sql]"
```

SQL logic tests live in `test/sql/` (`require mlx;`). Every GPU capability is diffed against
DuckDB's CPU results.

## License

MIT
