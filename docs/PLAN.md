# duckdb-mlx — GPU-Accelerated DuckDB on Apple Silicon (Metal / MLX)

**Status:** v0.3 — Phase 1 complete, early Phase 2 (GROUP BY v1, mixed-execution strategy)
**References:** NVIDIA GQE (rapidsai/gqe, NVIDIA blog Jun 2026), Sirius (sirius-db/sirius)
**North star:** *Best of both worlds* — DuckDB CPU for what it already wins; MLX GPU for
compute-dense, resident-data, and native-linear-algebra workloads at scale.

---

## 0. Progress Log

### 2026-07-02 (latest) — GROUP BY v1 + incremental dense cache + mixed-execution strategy

- **GROUP BY transparent interception:** `MLX_GROUPBY` / `MLX_GROUPBY_CACHED` for single
  integer key + `SUM(column)` on `seq_scan`, no `WHERE`. Optimizer reads output types
  from the aggregate plan; cache keys use storage column ids.
- **Dense perfect-hash path (O(n)):** `keys - min` → `scatter_add` into a fixed table —
  DuckDB `PERFECT_HASH_GROUP_BY` equivalent. Replaces the losing O(n log n) argsort path
  for low-cardinality groups.
- **Incremental GPU partial aggregates:** during cache population, segments scatter into a
  persistent dense table per `(group_col, value_col)` pair. Hot `MLX_GROUPBY_CACHED` reads
  ~G group rows, not N fact rows — **<1 ms vs CPU ~2–5 ms** on 5M rows / 1K groups (M4).
- **Cold path:** CPU double dense accumulate (accurate); GPU upload only when dense span
  exceeds host path or hash fallback needed.
- **Benchmark harness:** `benchmark/bench_minimal.sql` + `run_minimal.sh`; TPC-H runner
  at `benchmark/tpch/run.sh` (GQE 5-run hot methodology).
- **Measured (M4 base, minimal bench):** expression SUM 10M hot ~3 ms vs CPU ~19 ms
  (**~6×**); plain `sum(col)` parity (~2 ms); GROUP BY 5M hot **<1 ms** vs CPU ~5 ms;
  selective filtered multi-agg parity (compute-then-mask regression confirmed).
- **166 SQL differential assertions** across 10 test files (up from 109).
- **Strategy locked (§1.1):** do not intercept CPU-fast shapes; expand MLX scope
  methodically toward TPC-H via mixed CPU/GPU techniques (§6.4).
- **Next:** shape-aware cost decline, GROUP BY + WHERE + multi-agg (Q1), VARCHAR dict
  keys, late masking (Q6), `mlx_pin`, composite-key dense GROUP BY, hash join spike.

### 2026-07-02 (later) — Phase 1 shipped: transparent 14× via GPU-resident cache + fusion

- `OptimizerExtension` intercepts `AGGREGATE(SUM(expr)) ← [PROJECTION] ← SEQ_SCAN`
  (expressions: arithmetic, negate, sin/cos/sqrt/abs, casts, constants over
  DOUBLE/FLOAT/BIGINT/INTEGER; everything else declines untouched).
- **GPU-resident column cache** (§3.2 realized): first query populates fp32 column
  segments keyed `catalog.schema.table#col`; subsequent plans become `MLX_SUM_CACHED`,
  a pure source with **no table scan**. Row-count mismatch bypasses + repopulates.
  Populations are all-or-nothing per table so multi-column programs stay row-aligned.
- **Kernel fusion**: each segment's expression forest goes through `mx::compile`
  (§2 confirmed): unfused 70 ms → fused 13 ms. Gotcha: constant-valued outputs corrupt
  `mx::compile`'s output mapping — keep them host-side.
- **Measured (M4 base, 100M rows, hot)**: expression SUM 190 ms → 13 ms (**14×**);
  `sum(x)` 13 ms → 4 ms (fp32 bandwidth floor); cold 0.49 s; CPU nearly idle during GPU
  queries.
- **Coverage extension:** count/count(*)/avg/min/max on the shared IR, and
  WHERE clauses as GPU masks — both pushed-down table filters (translated + cleared
  from the scan, filter-only columns re-added to the projection) and residual filter
  nodes (comparisons, AND/OR/NOT). Counts exact, filtered expression queries ~2.8×
  (the CPU skips excluded rows' math; we compute-then-mask), multi-aggregate single
  pass 84 ms vs 115 ms. Gotcha: `table_filters` keys are storage column indexes, not
  `column_ids` positions.
- Segment-level zone-map pruning on cached columns (`mlx_cache_prune.test`).

### 2026-07-02 — Phase 0 complete + flagship use-case

**Done, all test-gated:**
- Extension skeleton from template; settings; spdlog via vcpkg; DuckDB pinned v1.5.4.
- MLX v0.31.2 vendored (`third_party/mlx`, built to `build/mlx-install`, imported
  target). Spike 1 (MLX-in-extension) and spike 2 (DuckDB→GPU data bridge) both **go**:
  `mlx_selftest()` = ok, `mlx_sum()` diffed against CPU over 1M rows.
- **Isolation constraint discovered:** DuckDB's vendored fmt shadows real fmt for all
  extension targets. Resolution: MLX and spdlog each live in a dedicated static lib;
  DuckDB headers and MLX/spdlog headers never share a TU. Permanent architecture of
  `bridge/`.
- **Flagship use-case shipped: GPU-resident vector similarity search** (`mlx_vss_pin`,
  `mlx_vss_search`, `mlx_vss_search_batch`). M4 (base, 24 GB), 1M×384 fp32: single
  query 3.5× (bandwidth floor), 32–128 batched queries **16–17×** vs DuckDB LATERAL
  brute force. fp16 pinning supported via `half` parameter.

**Measured platform facts (M4 base) that refine §2:**
- DuckDB CPU `sum()` runs at ~100 GB/s — full memory bandwidth. Scan-bound ops cannot
  win on GPU; **do not intercept them.**
- ALU-dense expressions (sin·cos+sqrt): GPU ~3–14× over CPU depending on scale and cache.
- The `list()` ingest vehicle costs 4–5× the GPU compute itself — transparent plan
  integration (Phase 1) is where end-to-end wins materialize.
- MLX `argpartition` appears sort-based on Metal (no gain over argsort at 1M): custom
  top-k kernel is a Tier-B candidate. Dense scatter beats sort for low-cardinality GROUP BY.
- MLX float64 is **not supported on GPU** — fp32 cache + scatter is the hot path;
  double accuracy on cold path via host accumulate.
- Metal open-addressing hash kernel (int64 keys) does not compile with device atomics —
  dense/scatter path is the production approach for low-cardinality; hash join spike TBD.

---

## 1. Vision

`duckdb-mlx` is a loadable DuckDB extension that **selectively** accelerates analytical
query pipelines on the Apple Silicon GPU via Metal and MLX. It ports the *winning concepts*
of NVIDIA GQE — GPU-friendly columnar layout, lightweight compression, zone-map pruning,
pipelined execution — and the *winning integration model* of Sirius — an optimizer hook
that intercepts supported DuckDB plans and falls back silently to CPU for everything else —
while removing every CUDA-native component.

The goal is not a GPU-only database. The goal is the **best of both worlds**: DuckDB's
mature CPU engine for scans, joins, and perfect-hash micro-kernels; MLX for fused
expression forests, resident fact-table analytics, incremental partial aggregates, and
native linear algebra (embeddings, matmul-heavy workloads).

```sql
LOAD 'duckdb_mlx';
SET mlx_execution = true;

-- GPU when shape + cost model say so (expression-heavy, cached, compute-dense):
SELECT sum(l_extendedprice * (1 - l_discount)) FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01';

-- CPU when DuckDB already wins (plain scan, join-heavy, tiny cardinality):
SELECT sum(l_quantity) FROM lineitem;  -- declines to CPU once shape-aware gate lands

SET mlx_execution = false;  -- per-connection opt-out
```

### 1.1 Best of Both Worlds — execution policy

Every query passes through a **capability gate** (can we translate it?) and a **cost
gate** (should we?). Declining to CPU is a feature, not a failure.

| Signal | Owner | Rationale |
|--------|-------|-----------|
| Plain `sum(col)`, `count(*)` on uncached table | **CPU** | ~100 GB/s memory bandwidth; GPU adds overhead |
| First-pass low-cardinality GROUP BY (small span) | **CPU** | DuckDB `PERFECT_HASH_GROUP_BY` is already ~2 ms |
| Expression / multi-agg over same scan | **GPU** | `mx::compile` fusion; one pass vs many CPU passes |
| Repeat query on GPU-resident cache | **GPU** | No scan; fused kernels over fp32 segments |
| Repeat GROUP BY on cached fact table | **GPU** | Incremental dense partial aggregate; download ~G rows |
| Highly selective filter + agg | **CPU** (today) | We compute-then-mask; CPU skips excluded rows |
| Multi-table joins, subqueries, EXISTS | **CPU** | Until hash join lands (Phase 3) |
| Embedding similarity / batched matmul | **GPU** | Native MLX strength; 16–17× demonstrated |
| ORDER BY + small LIMIT | **CPU** | Tiny output; sort on GPU rarely pays |

**Evolution principle:** expand MLX scope continuously across every row in §4, but only
intercept when the cost model predicts a win at realistic scale (large datasets, hot
cache, compute density). Never regress CPU-fast paths to chase benchmark headlines.

### Non-goals (v1)
- Multi-GPU / multi-node execution (irrelevant on Apple Silicon anyway).
- Windowing, ASOF joins, nested types, full VARCHAR expression coverage.
- Replacing DuckDB's storage format. We accelerate *execution over* DuckDB/Parquet data.
- Beating DuckDB on every query. Scan-bound queries share the same memory bandwidth as
  the CPU on Apple Silicon; the win is in compute-dense operators and resident caches.
- GPU-only TPC-H. Mixed execution (§6.4) is how we achieve strong aggregate numbers
  honestly.

---

## 2. Platform Reality Check: Grace Blackwell → Apple Silicon

GQE's headline gains come from *minimizing CPU→GPU data movement*. On Apple Silicon
**there is no discrete transfer**: CPU and GPU share one physical LPDDR pool. Every GQE
concept must be re-derived, not copied.

| GQE / Sirius (CUDA) | Purpose there | duckdb-mlx (Metal/MLX) replacement |
|---|---|---|
| cuDF relational operators | GPU kernels for join/agg/sort/filter | MLX ops (elementwise, reductions, scatter, gather) + **custom Metal kernels** via `mx.fast.metal_kernel` for hash tables, compact, decode |
| RMM memory pools | Sub-allocator over cudaMalloc | MLX unified allocator + `MTLHeap` scratch; `MTLResidencySet` (macOS 15+) |
| cudaMemcpyBatchAsync | Hide H2D latency | **Eliminated.** Layout conversion (DuckDB Vector → fp32 column) done lazily and cached |
| nvCOMP + Blackwell DE | Compress transfers | Cascaded-style GPU decode (Phase 2); skip LZ4-on-GPU in v1 |
| Substrait plan import | Decouple planner | **Not needed.** `OptimizerExtension` on logical plan (Sirius pattern) |
| Zone maps in GPU memory | Prune before transfer | DuckDB row-group stats + per-segment zone maps on GPU cache ✅ |
| 64-bit atomics | Hash table build | 32-bit atomics baseline; dense scatter for low-cardinality GROUP BY ✅ |

### Where the speedup actually comes from on Apple Silicon
1. **Compute-dense operators** — multi-expression aggregates, fused projection/filter
   trees. Measured **6–14×** on expression SUM (scale- and cache-dependent).
2. **Kernel fusion via `mx::compile`** — whole expression forests in few kernels.
3. **Resident GPU cache + incremental partial state** — repeat queries skip scan;
   dense GROUP BY table built during cache populate. Measured **<1 ms** hot GROUP BY.
4. **Native linear algebra** — VSS batched matmul **16–17×**.
5. **Pruning** — segment zone maps skip work before kernel launch.

### Where we do not win (confirmed — stay on CPU)
- Plain `sum(column)` — parity (~2 ms / 10M rows).
- Selective filter + agg with compute-then-mask — parity vs CPU row skipping.
- Cold first query — always slower (scan + cache build + first scatter).
- Join-heavy TPC-H queries — CPU until Phase 3.

Honest expectation: **5–10× on GPU-amenable TPC-H queries** (Q1, Q6, Q14) at SF1 hot
cache; **parity** on scan-bound queries; **aggregate suite speedup** computed over the
GPU-amenable subset only (§7).

---

## 3. Architecture

```
            ┌────────────────────────────────────────────────┐
            │                  DuckDB core                    │
            │  parser → binder → optimizer → physical plan    │
            └───────────────┬────────────────────────────────┘
                            │ OptimizerExtension hook (Sirius pattern)
            ┌───────────────▼────────────────────────────────┐
 QUERY      │  Plan Interceptor + Cost Model (§1.1)            │
 LAYER      │  • capability: can we translate this subtree?    │
            │  • cost: will GPU beat CPU at this scale/shape?  │
            │  • emit MLX physical op or decline → CPU         │
            └───────────────┬────────────────────────────────┘
            ┌───────────────▼────────────────────────────────┐
 DATA       │  GPU Column Cache ("MlxTableFormat" lite) ✅     │
 LAYER      │  • fp32 segments keyed catalog.schema.table#col  │
            │  • zone maps per segment; incremental populate   │
            │  • incremental dense GROUP BY accumulators ✅    │
            │  • [planned] cascaded encode, dict strings, LRU  │
            └───────────────┬────────────────────────────────┘
            ┌───────────────▼────────────────────────────────┐
 EXECUTION  │  MLX/Metal Operator Runtime                      │
 LAYER      │  • Tier A: project, filter, reduce, scatter ✅   │
            │  • Tier A: mx::compile fusion ✅                 │
            │  • Tier B: dense groupby scatter ✅              │
            │  • Tier B: hash join, compact, decode [planned]  │
            └────────────────────────────────────────────────┘
```

### 3.1 Query layer — plan interception

**Implemented ✅**
- `OptimizerExtension` on logical plan; greedy intercept of supported subtrees.
- Physical operators: `MLX_SUM`, `MLX_SUM_CACHED`, `MLX_GROUPBY`, `MLX_GROUPBY_CACHED`.
- Settings: `mlx_execution`, `mlx_min_rows` (default 512K), `mlx_log_level`.
- Translate-time fallback: unsupported shape → plan untouched (CPU runs).

**In progress 🟡**
- Shape-aware cost decline (plain SUM, first-pass GROUP BY where CPU wins).
- GROUP BY + WHERE; multi-aggregate GROUP BY (Q1).
- Composite-key packing for multi-column low-cardinality GROUP BY.

**Planned ❌**
- Hybrid CPU/GPU plans (GPU fact-table subtree → CPU join parent).
- Runtime fallback: OOM / kernel failure → transparent CPU re-execution.
- `mlx_pin('table')` explicit pre-warm.

### 3.2 Data layer — MlxTableFormat

**Implemented ✅**
- fp32 column segments, row-aligned incremental population.
- Cache keys `catalog.schema.table#storage_col_id`; population tracking.
- Per-segment zone maps; prune-before-execute on cached WHERE.
- Process-lifetime cache singleton; `mlx_cache_clear()` for tests.
- Incremental dense GROUP BY accumulator per `(group_key, value_key)`.

**Planned ❌**
- Zero-copy fast path (`newBufferWithBytesNoCopy` for aligned vectors).
- Cascaded encoding (delta → RLE → bit-pack) + fused decode kernel.
- Dictionary-encoded VARCHAR columns.
- LRU eviction / `mlx_max_cache_memory` budget.
- fp64 value segments for exact SUM on hot path (or host-side double accumulators).

### 3.3 Execution layer — operators

**Tier A — MLX graph ops (implemented ✅ unless noted)**
- Projection & scalar expressions (arithmetic, sin/cos/sqrt/abs, casts, comparisons).
- Filter as GPU mask (compute-then-mask — late compact planned).
- Ungrouped agg: sum/count/avg/min/max; multi-agg single fused pass.
- `mx::compile` per cache segment.

**Tier B — custom Metal / dense paths**
| Kernel | Status | Notes |
|--------|--------|-------|
| Dense GROUP BY scatter | ✅ | O(n) perfect-hash; hot path for low cardinality |
| Incremental dense accumulate | ✅ | Built during cache populate |
| `groupby_hash` (open addressing) | ❌ | Metal atomic assignment errors; not production |
| `hash_build` / `hash_probe` | ❌ | Phase 3 join spike |
| `selection_compact` (late mask) | ❌ | Phase 2 — fixes Q6 selective regression |
| `cascaded_decode` | ❌ | Phase 2 |
| `dict_gather_strings` | ❌ | Phase 3 |
| Custom VSS top-k | ❌ | Phase 2 stretch |

---

## 4. Operator & Type Coverage Matrix

Legend: ✅ done · 🟡 partial · ❌ not started · — decline to CPU by design

| | INT32/64 | FLOAT/DOUBLE | DECIMAL(≤18) | DATE/TIMESTAMP | VARCHAR (dict) |
|---|---|---|---|---|---|
| Filter / Projection | ✅ | ✅ | ❌ | 🟡 epoch via double | ❌ |
| Ungrouped agg | ✅ | ✅ | ❌ | 🟡 min/max | — |
| GROUP BY hash agg | 🟡 1 key + SUM | 🟡 1 key + SUM | ❌ | ❌ | ❌ |
| GROUP BY multi-agg | ❌ | ❌ | ❌ | ❌ | ❌ |
| Hash JOIN | ❌ | ❌ | ❌ | ❌ | ❌ |
| ORDER BY / TOP-N | ❌ | ❌ | ❌ | ❌ | ❌ |
| VSS / embeddings | — | ✅ fp32/fp16 | — | — | — |

**MLX scope evolution (expand continuously, intercept selectively):**

| Phase | Coverage expansion |
|-------|-------------------|
| **2a (now)** | GROUP BY + WHERE; multi-agg GROUP BY; VARCHAR dict keys; late masking; shape-aware decline; composite keys |
| **2b** | ORDER BY / TOP-N on GPU; cascaded cache encoding; `mlx_pin`; fp64 accum option |
| **3** | Hash join; hybrid fact+dim plans; string dict end-to-end; semi/anti mark join |
| **4** | Memory budget; runtime CPU fallback; `mlx_stats()`; community extension packaging |
| **5+** | WINDOW; DECIMAL128; Parquet-direct GPU decode; hybrid plan bridges; ANE experiments |

Everything outside the matrix — WINDOW, nested types, LIST/STRUCT, regex, LIKE beyond
prefix — declines to CPU until explicitly added.

---

## 5. Repo & Build

**Current layout** (monolithic bridge phase; module split deferred):

```
duckdb-mlx/
├── src/
│   ├── duckdb_mlx_extension.cpp   # load/register, settings, scalar functions
│   ├── planner/mlx_transparent.cpp # OptimizerExtension + physical operators
│   ├── bridge/
│   │   ├── mlx_bridge.cpp         # cache, fusion, VSS bridge, zone maps
│   │   └── mlx_groupby.cpp        # dense scatter, incremental accumulators
│   ├── ops/mlx_vss.cpp            # VSS table functions
│   ├── include/                   # mlx_bridge.hpp, mlx_groupby_detail.hpp, …
│   ├── format/                    # .gitkeep — cache logic lives in bridge/ for now
│   ├── kernels/                   # .gitkeep — inline mx.fast.metal_kernel for now
│   └── exec/                      # .gitkeep
├── test/sql/                      # 10 sqllogictest files, 166 assertions
├── benchmark/
│   ├── bench_minimal.sql          # quick CPU vs GPU snapshot
│   ├── run_minimal.sh
│   ├── bench_transparent.sql      # 100M expression SUM
│   ├── bench_groupby.sql
│   └── tpch/                      # GQE-style 22-query harness
└── docs/PLAN.md
```

- Language: C++17 linking `mlx::core`; Metal via `mx.fast.metal_kernel` inline.
- Platform: macOS 14+ arm64; CI on GitHub Actions macos arm64 runners.
- Build: `GEN=ninja make release`; tests: `./build/release/test/unittest "[sql]"`.
- Benchmark: `./benchmark/run_minimal.sh` or `./benchmark/tpch/run.sh 1`.

---

## 6. Phased Roadmap

### Phase 0 — Feasibility spikes ✅ Complete

| Spike | Result |
|-------|--------|
| MLX-in-extension | ✅ Go |
| Zero-copy bridge | 🟡 Partial — fp32 densify default; nocopy path not implemented |
| Hash table kernel | ❌ Join spike not run; groupby Metal hash blocked by atomics |
| Cascaded decode | ❌ Not started |
| Threading | 🟡 Implicit via DuckDB pipeline; no contention study |

### Phase 1 — Skeleton & first transparent query ✅ Complete

- Optimizer hook; `SCAN → FILTER → PROJ → UNGROUPED AGG` on numeric columns.
- GPU column cache; `MLX_SUM_CACHED`; `mx::compile` fusion.
- Multi-agg, WHERE masks, zone-map pruning.
- Differential test harness (permanent).
- VSS flagship (fp16 option).
- *Milestone:* Q6-shaped queries pass (`tpch_shapes.test`). Real TPC-H Q6 on SF1: pending
  late masking for selective win.

### Phase 2 — Group-by, masking, TPC-H fact-table path 🟡 In progress (~30%)

**2a — Immediate (TPC-H fact-table wins)**
- [ ] Shape-aware cost decline (don't intercept plain SUM / CPU-fast GROUP BY cold).
- [ ] GROUP BY + WHERE (mask → compact → dense scatter).
- [ ] Multi-aggregate GROUP BY (Q1: 6 aggs in one fused pass).
- [ ] VARCHAR dictionary keys (`l_returnflag`, `l_linestatus`).
- [ ] Composite-key dense GROUP BY (pack multi-column low-cardinality keys).
- [ ] Late masking / `selection_compact` (Q6 selective speedup).
- [ ] `mlx_pin('lineitem')` pre-warm for benchmark suite.

**2b — Scale & encoding**
- [ ] Cascaded encoding + fused decode in cache pipeline.
- [ ] Partition pipelining across MLX streams.
- [ ] Custom VSS top-k Metal kernel.
- [ ] Per-column-set cache populations (fix all-or-nothing thrash).
- [ ] Cache eviction / `mlx_max_cache_memory`.

*Milestones:*
- TPC-H Q1 + Q6 correct and **5–10× hot** at SF1.
- GPU-amenable subset aggregate **≥5×** (GQE methodology, §7).
- Cache-resident re-runs ≥2× vs DuckDB hot CPU on expression workloads ✅ (already 6–14×).

### Phase 3 — Joins, hybrid plans, strings ❌ Not started

- `hash_build` / `hash_probe`; mark join for semi/anti.
- **Hybrid execution:** GPU lineitem filter+project+cache → CPU hash join to dimensions.
- Dictionary VARCHAR end-to-end.
- Runtime CPU fallback on GPU failure.
- *Milestone:* ≥10 of 22 TPC-H queries use GPU for fact-table portion at SF10; join
  queries via hybrid decomposition.

### Phase 4 — Hardening & release ❌ Not started

- Memory budget, concurrency, `mlx_stats()`, community extension packaging.
- Blog + reproduction scripts.

### Phase 5+ — Stretch

- Full hybrid CPU/GPU plan bridges.
- Parquet-direct GPU decode.
- WINDOW, DECIMAL128, ANE filter experiments.
- Optional Substrait consumer for GQE-comparable standalone benchmarking.

### 6.4 TPC-H mixed-execution strategy

Goal: **incredible aggregate numbers honestly** — GPU for fact-table analytics, CPU for
joins and CPU-fast shapes. Do not dilute the speedup metric with queries we never intend
to accelerate.

```
Session / benchmark start
  └─ mlx_pin('lineitem')  [planned] — one-time cold cost, amortized across suite

Per query:
  ├─ GPU:  lineitem scan shapes (Q1, Q6, Q12, Q14, Q19)
  │         fused multi-agg, cached, dense GROUP BY, late mask
  ├─ HYBRID: GPU fact projection + CPU join (Q3, Q5, Q7+)  [Phase 3]
  └─ CPU:  joins, subqueries, EXISTS, ORDER BY small LIMIT (Q2, Q4, Q15–Q22)
```

| Query | Primary work | Target engine | Phase |
|-------|-------------|---------------|-------|
| **Q1** | lineitem filter + 6 aggs + 2-key GROUP BY | **GPU** | 2a |
| **Q6** | selective filter + expression SUM | **GPU** (late mask) | 2a |
| **Q12** | join + CASE GROUP BY | GPU lineitem / CPU join | 2b/3 |
| **Q14** | join + conditional SUM ratio | **GPU** expr agg | 2a |
| **Q19** | OR filter + expression SUM | **GPU** | 2a |
| Q3, Q5, Q7–Q10 | multi-join + GROUP BY | HYBRID | 3 |
| Q2, Q4, Q11, Q13, Q15–Q22 | joins, subqueries, EXISTS | **CPU** | — |

**Benchmark reporting (§7):** tag each query `GPU` | `HYBRID` | `CPU`; report aggregate
speedup over GPU-tagged queries only; always report cold separately.

**Target (SF1, hot cache, M4 class):**

| Query | CPU (est.) | GPU hot (target) | Speedup |
|-------|------------|------------------|---------|
| Q1 | 200–400 ms | 30–60 ms | 5–10× |
| Q6 | 50–80 ms | 5–10 ms | 6–10× |
| Q14 | ~100 ms | ~20 ms | ~5× |
| GPU subset aggregate | — | — | **5–8×** |

---

## 7. Benchmarking Plan

**Quick snapshot:** `./benchmark/run_minimal.sh` — expression SUM, plain SUM, GROUP BY,
filtered multi-agg at 10M/5M rows.

**TPC-H:** `./benchmark/tpch/run.sh 1` — 22 queries, 5-run hot average, cold reported
separately. Runner marks unsupported queries; classify as GPU/HYBRID/CPU per §6.4.

**Methodology (GQE-aligned):**
- Same-machine DuckDB CPU vs duckdb-mlx GPU.
- 5 hot-cache runs averaged; cold (ingest + first scatter) reported separately.
- Aggregate speedup = Σ(CPU_GPU_queries) / Σ(GPU_hot_GPU_queries) — not all 22.
- Per-query attribution via `mlx_cache_stats()` (segments pruned); kernel timing TBD.
- Hardware: M4 base (current dev), M3/M4 Max (target), report GB/s-normalized where useful.
- Standard TPC disclaimer (non-audited, non-comparable).

**Current minimal bench (M4 base, Jul 2026):**

| Workload | CPU | GPU hot | Speedup |
|----------|-----|---------|---------|
| Expression SUM 10M | ~19 ms | ~3 ms | ~6× |
| Plain SUM 10M | ~2 ms | ~2 ms | ~1× (stay CPU) |
| GROUP BY 5M / 1K groups | ~5 ms | <1 ms | ~5–10× |
| Filtered count+sum 10M | ~6 ms | ~6 ms | ~1× (fix with late mask) |

---

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Scan-bound queries show no win | High | Shape-aware decline (§1.1); don't benchmark as GPU targets |
| Compute-then-mask loses on selective filters | High | Late masking / compact before agg (Phase 2a) |
| fp32 cache accuracy drift | Medium | Host double on cold path; fp64 accum option; tolerance in tests |
| Metal hash atomics unusable for int64 keys | Confirmed | Dense scatter for low-cardinality; redesign hash for 32-bit slots |
| MLX float64 not on GPU | Confirmed | fp32 GPU + double host accumulate |
| Cold query tax hurts first-run UX | High | `mlx_pin`; amortize across session; report hot separately |
| All-or-nothing cache populations thrash | Medium | Per-column-set populations (Phase 2b) |
| DuckDB API churn | Medium | Pin submodule; Sirius proves hook surface |
| Correctness divergence | High-impact | 166 differential assertions; expand with each coverage row in §4 |

---

## 9. Open Questions

| # | Question | Resolution |
|---|----------|------------|
| 1 | MLX C++ vs mlx-c? | ✅ C++ via isolated `bridge/` static lib |
| 2 | Logical vs physical plan? | ✅ Logical (`OptimizerExtension`) |
| 3 | Partition size? | 🟡 Ad hoc segment sizes from DuckDB vectors; tune in 2b |
| 4 | Cache persists across connections? | ✅ Process-lifetime singleton |
| 5 | Decimal as scaled int64? | ❌ Fallback for now |
| 6 | Explicit `mlx_query()` API? | 🟡 Scalar/bench APIs exist; full SQL passthrough deferred |
| 7 | When to decline to CPU on cost? | 🟡 §1.1 policy defined; implementation in 2a |
| 8 | TPC-H aggregate over all 22 vs GPU subset? | ✅ GPU-subset only (§6.4, §7) |

---

## 10. Concept Traceability (GQE/Sirius → duckdb-mlx)

| Winner concept | Source | Ported as | Status |
|----------------|--------|-----------|--------|
| Optimizer-hook interception + CPU fallback | Sirius | §3.1 | ✅ translate-time |
| Extension-template packaging | Sirius | §5 | ✅ |
| In-memory table format / GPU cache | GQE | §3.2 | ✅ fp32 lite |
| Kernel fusion | GQE/MLX | `mx::compile` | ✅ |
| Zone-map pruning | GQE | Segment zone maps | ✅ |
| Perfect-hash GROUP BY | GQE | Dense scatter + incremental accum | ✅ hot path |
| Pipelined decode∥compute | GQE | MLX streams | ❌ Phase 2b |
| Cascaded compression | GQE | Cascaded encode/decode | ❌ Phase 2b |
| Hash join + mark join | GQE | Tier B kernels | ❌ Phase 3 |
| Hybrid CPU/GPU plans | Sirius/GQE | Fact GPU + dim CPU | ❌ Phase 3 |
| Hot-cache 5-run TPC-H methodology | GQE | §7 | ✅ harness |
| Best-of-both-worlds cost model | duckdb-mlx | §1.1 | 🟡 policy set; code in 2a |

---

## 11. Current Scorecard

```
Phase 0  ████████████████████  100%
Phase 1  ████████████████████  100%
Phase 2  ██████░░░░░░░░░░░░░░   30%  ← GROUP BY v1, dense cache, zone prune done
Phase 3  ░░░░░░░░░░░░░░░░░░░░    0%
Phase 4  ░░░░░░░░░░░░░░░░░░░░    0%

Tests:     166 assertions / 10 SQL files
Operators: MLX_SUM[_CACHED], MLX_GROUPBY[_CACHED]
Flagship:  VSS 16–17× batched; expression SUM 6–14× hot; GROUP BY <1 ms hot
Next:      Q1 path (multi-agg + VARCHAR + WHERE), Q6 late mask, shape-aware decline
```

**One-liner:** CPU owns bandwidth and joins; MLX owns fused analytics on resident fact
tables — expand scope continuously, intercept selectively, benchmark honestly.
