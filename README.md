# duckdb-mlx

GPU-accelerated DuckDB on Apple Silicon (Metal / MLX).

`duckdb_mlx` is a loadable DuckDB extension that transparently executes analytical query
pipelines on the Apple Silicon GPU, and falls back silently to DuckDB's CPU engine for
anything unsupported. See [docs/PLAN.md](docs/PLAN.md) for the full architecture and
roadmap; the design ports concepts from NVIDIA GQE and Sirius, re-derived for unified
memory.

```sql
LOAD 'duckdb_mlx';
-- plain SQL now runs on the Apple GPU when supported
SELECT l_returnflag, sum(l_quantity) FROM lineitem GROUP BY 1 ORDER BY 1;
SET mlx_execution = false;  -- per-connection opt-out
```

**Status:** Phase 0 — extension skeleton. The extension loads, registers its settings
(`mlx_execution`, `mlx_min_rows`, `mlx_log_level`) and an `mlx_info()` function; no GPU
execution yet.

The extension builds on all DuckDB platforms. The GPU execution path is compiled in only
on Apple Silicon (`DUCKDB_MLX_ENABLE_GPU`, auto-detected); elsewhere every plan stays on
the CPU engine.

## Repository layout

```
src/
├── duckdb_mlx_extension.cpp   # load/register, settings, optimizer hook
├── planner/                   # logical-plan walker, supported-subtree detection
├── format/                    # MlxTableFormat: ingest, encode, zone maps, cache
├── ops/                       # Tier A graph builders (project/filter/agg/sort)
├── kernels/                   # Tier B .metal sources + launch wrappers
├── exec/                      # pipeline scheduler, streams, fallback runtime
└── bridge/                    # DataChunk ⇄ GPU buffer conversion
test/sql/                      # sqllogictest files (DuckDB harness)
test/cpp/                      # kernel unit tests
benchmark/                     # TPC-H harness, GPU-vs-CPU diff runner
```

## Building

### Dependencies

C++ dependencies are managed with [vcpkg](https://vcpkg.io) for cross-platform support
(declared in `vcpkg.json`, currently spdlog). Set it up once:

```shell
git clone https://github.com/Microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh
export VCPKG_TOOLCHAIN_PATH=`pwd`/vcpkg/scripts/buildsystems/vcpkg.cmake
```

MLX itself is Apple-only and not in the vcpkg registry; it will be vendored under
`third_party/` when the GPU execution path lands (Phase 0 spike).

### Build steps

```shell
git submodule update --init
make
```

The main binaries that will be built are:

```shell
./build/release/duckdb                                            # shell with the extension pre-loaded
./build/release/test/unittest                                     # test runner
./build/release/extension/duckdb_mlx/duckdb_mlx.duckdb_extension  # loadable extension
```

## Running

```
$ ./build/release/duckdb
D SELECT mlx_info();
┌─────────────────────────────────────────┐
│               mlx_info()                │
│                 varchar                 │
├─────────────────────────────────────────┤
│ duckdb_mlx gpu=available spdlog=1.16.0  │
└─────────────────────────────────────────┘
```

## Testing

SQL logic tests live in `./test/sql` and run with:

```shell
make test
```

Every GPU-supported query will also be executed with `mlx_execution=false` and the
results diffed automatically — this differential harness is permanent infrastructure
(see docs/PLAN.md §Phase 1).

## License

MIT
