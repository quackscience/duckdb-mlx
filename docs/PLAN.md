# duckdb-mlx — GPU-Accelerated DuckDB on Apple Silicon (Metal / MLX)

**Status:** v0.2 — Phase 0 complete, flagship use-case shipped (see §0)
**References:** NVIDIA GQE (rapidsai/gqe, NVIDIA blog Jun 2026), Sirius (sirius-db/sirius)

---

## 0. Progress Log

### 2026-07-02 — Phase 0 complete + flagship use-case

**Done, all test-gated (26 sqllogictest assertions passing):**
- Extension skeleton from template; settings; spdlog via vcpkg; DuckDB pinned v1.5.4.
- MLX v0.31.2 vendored (`third_party/mlx`, built to `build/mlx-install`, imported
  target). Spike 1 (MLX-in-extension) and spike 2 (DuckDB→GPU data bridge) both **go**:
  `mlx_selftest()` = ok, `mlx_sum()` diffed against CPU over 1M rows.
- **Isolation constraint discovered:** DuckDB's vendored fmt (namespace `duckdb_fmt`,
  same `<fmt/...>` include paths/guards) shadows real fmt for all extension targets.
  Resolution: MLX and spdlog each live in a dedicated static lib with inherited include
  dirs cleared; DuckDB headers and MLX/spdlog headers never share a TU. This is now the
  permanent architecture of `bridge/` (and validates §9 Q1: C++ core works, no
  exception/ABI friction observed).
- **Flagship use-case shipped: GPU-resident vector similarity search** (`mlx_vss_pin`,
  `mlx_vss_search`, `mlx_vss_search_batch`) — the §3.2 resident-cache concept
  specialized to embeddings. M4 (base, 24 GB), 1M×384 fp32: single query 3.5×
  (bandwidth floor), 32–128 batched queries **16–17×** vs DuckDB LATERAL brute force.

**Measured platform facts (M4 base) that refine §2:**
- DuckDB CPU `sum()` runs at ~100 GB/s — full memory bandwidth. Confirmed: scan-bound
  ops cannot win on GPU; do not chase them.
- ALU-dense expressions (sin·cos+sqrt): GPU ~3× over CPU net of ingest overhead.
- The `list()` ingest vehicle costs 4–5× the GPU compute itself — direct plan/scan
  integration (Phase 1) is where end-to-end wins materialize.
- MLX `argpartition` appears sort-based on Metal (no gain over argsort at 1M): a custom
  top-k kernel is a real Tier-B candidate.

**Next, in effort-to-payoff order:**
1. fp16 pinning for VSS (halves bytes → ~2× across the board; standard for embeddings).
2. `mlx_vss_pin` from a table scan directly (drop the `list()` vehicle).
3. Custom top-k Metal kernel (large-Q batches leave time on the table).
4. Phase 1 optimizer hook (transparent `SCAN→FILTER→PROJ→AGG`), per original roadmap.

---

## 1. Vision

`duckdb-mlx` is a loadable DuckDB extension that transparently executes analytical query
pipelines on the Apple Silicon GPU via Metal and MLX. It ports the *winning concepts* of
NVIDIA GQE — GPU-friendly columnar layout, per-column lightweight compression with GPU
decompression, zone-map partition pruning, pipelined chunked execution — and the *winning
integration model* of Sirius — an optimizer hook that intercepts DuckDB plans and falls
back silently to CPU for anything unsupported — while removing every CUDA-native component.

```sql
LOAD 'duckdb_mlx';
-- plain SQL now runs on the Apple GPU when supported
SELECT l_returnflag, sum(l_quantity) FROM lineitem GROUP BY 1 ORDER BY 1;
SET mlx_execution = false;  -- per-connection opt-out
```

### Non-goals (v1)
- Multi-GPU / multi-node execution (irrelevant on Apple Silicon anyway).
- Windowing, ASOF joins, nested types, full VARCHAR expression coverage.
- Replacing DuckDB's storage format. We accelerate *execution over* DuckDB/Parquet data.
- Beating DuckDB on every query. Scan-bound queries share the same memory bandwidth as
  the CPU on Apple Silicon; the win is in compute-dense operators (joins, group-bys,
  sorts, expression-heavy projections) and in keeping working sets GPU-resident.

---

## 2. Platform Reality Check: Grace Blackwell → Apple Silicon

This is the most important section. GQE's headline gains come from *minimizing and hiding
CPU→GPU data movement* (NVLink-C2C, Blackwell Decompression Engine, cudaMemcpyBatchAsync).
On Apple Silicon **there is no discrete transfer**: CPU and GPU share one physical LPDDR
pool behind one memory controller. Every GQE concept must be re-derived, not copied.

| GQE / Sirius (CUDA) | Purpose there | duckdb-mlx (Metal/MLX) replacement |
|---|---|---|
| cuDF relational operators | GPU kernels for join/agg/sort/filter | MLX ops where they map cleanly (elementwise, reductions, argsort, take/gather, scan) + **custom Metal kernels** via `mx.fast.metal_kernel` / raw MSL for hash tables, radix partitioning, string gather |
| RMM memory pools | Sub-allocator over cudaMalloc | MLX unified allocator + `MTLHeap` for kernel scratch; `MTLResidencySet` (macOS 15+) to keep hot buffers wired |
| cudaMemcpyBatchAsync + pinned bounce buffers | Hide H2D latency | **Eliminated.** Zero-copy: wrap page-aligned host columns with `newBufferWithBytesNoCopy` / MLX arrays over shared memory. "Transfer" becomes *layout conversion* (DuckDB Vector → dense GPU-friendly column), done lazily and cached |
| nvCOMP LZ4 + Cascaded, Blackwell Decompression Engine | Compress transfers; capacity | **Cascaded-style GPU decode in Metal** (delta + RLE + bit-pack are embarrassingly parallel and trivially expressible). Skip LZ4-on-GPU in v1 — no DE hardware, byte-serial LZ77 decode is a poor fit for Apple GPUs; rely on DuckDB's own lightweight storage encodings + our cascaded format for the GPU cache |
| CUDA streams, per-thread default stream | Overlap transfer/decode/compute | MLX streams + Metal command queues. Pipeline stages become: *materialize chunk → decode → compute*, overlapped across DuckDB row groups; far fewer stages needed since stage "H2D" is gone |
| Substrait plan import (DataFusion producer) | Decouple planner from engine | **Not needed as ingress.** Hook DuckDB's `OptimizerExtension` directly on the logical plan (Sirius pattern) — no substrait extension dependency, no plan serialization round-trip. (Optional substrait consumer later for standalone benchmarking parity with GQE) |
| io_uring custom Parquet reader | Saturate NVMe → GPU | DuckDB's Parquet reader + our GPU column cache. Apple NVMe + unified memory makes a custom reader low ROI for v1 |
| Zone maps as cuDF tables in GPU memory | Prune before transfer | Same concept, two tiers: (a) **reuse DuckDB's existing row-group zone maps** during plan binding; (b) build partition-level min/max as MLX arrays for GPU-cached tables so pruning of the *GPU cache* is itself a GPU kernel |
| CUDA cooperative groups, 64-bit atomics | Hash table build | Metal SIMD-group ops (width 32), threadgroup memory (32 KB), **32-bit atomics only** as the portable baseline → design hash tables around 32-bit slots + tag/verify, or Apple-family-9 (M3/M4) 64-bit atomic paths behind a feature flag |

### Where the speedup actually comes from on Apple Silicon
1. **Compute-dense operators.** Hash joins, grouped aggregation with many groups,
   multi-key sorts, expression-heavy filters — GPU ALU throughput and latency-hiding
   beat P-cores even at equal bandwidth.
2. **Kernel fusion via MLX lazy evaluation.** Whole projection/filter expression trees
   compile into few kernels (`mx.compile`), vs DuckDB's vector-at-a-time interpretation.
3. **Resident compressed GPU cache.** Cascaded-encoded columns stay resident; repeated
   analytical queries skip Parquet decode and DuckDB scan entirely (this is the analog of
   GQE's "in-memory table format", and where GQE-like multiples become plausible).
4. **Pruning.** Same 1.4×-class end-to-end win GQE measured, cheap to get by reusing
   DuckDB metadata.

Honest expectation setting: TPC-H Q1/Q6-style scans → parity to ~2×; join/agg-heavy
queries on M3/M4 Max class → 2–6× is the realistic target band, not GQE's 25× (which was
one B200 vs CPU over NVLink with a hardware decompression engine).

---

## 3. Architecture

Same three-layer decomposition as GQE, re-derived for unified memory.

```
            ┌────────────────────────────────────────────────┐
            │                  DuckDB core                    │
            │  parser → binder → optimizer → physical plan    │
            └───────────────┬────────────────────────────────┘
                            │ OptimizerExtension hook (Sirius pattern)
            ┌───────────────▼────────────────────────────────┐
 QUERY      │  Plan Interceptor & Translator                 │
 LAYER      │  • walk LogicalOperator tree                   │
            │  • supported-subtree detection (greedy maximal)│
            │  • emit MlxPipeline plan or decline → CPU      │
            └───────────────┬────────────────────────────────┘
            ┌───────────────▼────────────────────────────────┐
 DATA       │  GPU Table Cache ("MlxTableFormat")            │
 LAYER      │  • row groups → fixed partitions (default 1M)  │
            │  • per-column cascaded encoding (Δ, RLE, pack) │
            │  • dictionary-encoded strings                  │
            │  • zone maps (min/max per partition, MLX array)│
            │  • zero-copy DataChunk ingestion path          │
            └───────────────┬────────────────────────────────┘
            ┌───────────────▼────────────────────────────────┐
 EXECUTION  │  MLX/Metal Operator Runtime                    │
 LAYER      │  • MLX ops: project, filter, reduce, sort      │
            │  • MSL kernels: hash build/probe, groupby,     │
            │    radix partition, decode, selection compact  │
            │  • stream-pipelined over partitions            │
            │  • result → DuckDB DataChunks (zero/low copy)  │
            └────────────────────────────────────────────────┘
```

### 3.1 Query layer — plan interception
- Register an `OptimizerExtension` (and/or replace physical plan via
  `duckdb::Extension` hooks) exactly as Sirius does: inspect the optimized logical plan,
  find the **maximal supported subtree** rooted as close to the sink as possible.
- v1 policy: all-or-nothing per query (simpler, Sirius v1 did the same); v2: hybrid
  plans where a GPU subtree feeds a CPU parent through a bridge operator.
- Cost gate: skip GPU for tiny inputs (estimated cardinality below threshold, e.g.
  < 512K rows scanned) — kernel launch + graph eval overhead dominates small queries.
- `SET mlx_execution = true|false`, `SET mlx_min_rows = N`, `SET mlx_log_level = ...`.
- **Fallback is sacred:** any unsupported type/operator/expression at translate time →
  return the plan untouched. Any runtime error (OOM, kernel failure) → re-execute on CPU
  and log. Correctness beats acceleration, always.

### 3.2 Data layer — MlxTableFormat (the GQE in-memory table, unified-memory edition)
- **Granularity:** table → row groups (align with DuckDB's 122,880-row row groups where
  sourcing from native storage; align with Parquet row groups when sourcing Parquet) →
  fixed-size partitions (tunable, default 1M rows) as the pruning/pipelining unit.
- **Ingestion paths:**
  1. *Cold path:* DuckDB table scan feeds DataChunks; we densify + encode into the cache
     on first GPU query (amortized, like GQE's ~1% load-time zone-map cost).
  2. *Zero-copy fast path:* for uncompressed, non-null fixed-width DuckDB vectors that
     are page-aligned, wrap directly (`newBufferWithBytesNoCopy`) without densify.
  3. *Explicit:* `CALL mlx_pin('lineitem')` to pre-encode and wire a table.
- **Encodings (per column, chosen by the GQE heuristic pair):** try cascaded
  (delta → RLE → bit-pack) first, measure ratio; fall back to plain. Ratio thresholds
  configurable (`mlx_compression_ratio_threshold`, default 1.0, mirroring GQE knobs).
  Compression here buys *capacity* (bigger-than-DRAM working sets stay cached) and
  *bandwidth* (scan reads fewer bytes; decode is ALU-cheap on GPU).
- **Strings:** global-per-column dictionary encoding; joins/group-bys operate on codes;
  late materialization of payload strings only at the sink. Non-dictionary-friendly
  VARCHAR columns disqualify the operator subtree in v1 (fallback).
- **NULLs:** validity bitmask per partition, Arrow-compatible layout, so a future
  Arrow/cuDF-interchange story is cheap.
- **Zone maps:** per-partition min/max stored as small MLX arrays; pruning expression is
  derived from the translated filter predicates and evaluated in one tiny kernel over
  all partitions (GQE measured ~2.2 ms overhead at 1 TB — ours will be microseconds at
  our scale).
- **Eviction:** LRU over partitions with a memory budget
  (`mlx_max_cache_memory`, default e.g. 50% of `recommendedMaxWorkingSetSize`);
  compressed partitions evict to "decoded-dropped" state first, then fully.

### 3.3 Execution layer — operators
Two-tier kernel strategy:

**Tier A — pure MLX graph ops** (get lazy fusion + `mx.compile` for free):
- Projection & scalar expressions (arithmetic, comparisons, boolean logic, CASE via
  `mx.where`, casts, date extract via integer math on epoch days).
- Filter: predicate → boolean mask → `mx.compact`/gather by indices
  (selection vectors as index arrays).
- Ungrouped aggregation: `sum/min/max/mean/count` are single reductions.
- ORDER BY / TOP-N: `mx.argsort` (+ composite key packing for multi-column),
  `mx.topk` for LIMIT-with-ORDER.

**Tier B — custom Metal kernels** (via `mx.fast.metal_kernel` or a small MSL library
loaded alongside; these are the port of what cuDF/cuCollections provided):
1. `hash_build` / `hash_probe` — open-addressing, linear probing, 32-bit atomic CAS
   slots with key-tag verification; bucket-chained overflow for multi-match inner joins.
2. `groupby_hash` — same table, atomic accumulate for sum/count/min/max;
   two-phase (partition → per-partition table) when group cardinality is high.
3. `radix_partition` — for partitioned join/agg when hash table exceeds a size budget.
4. `cascaded_decode` — fused bit-unpack + RLE-expand + delta-scan (prefix sum) kernel.
5. `selection_compact` — stream compaction via SIMD-group ballot + prefix sum.
6. `dict_gather_strings` — late string materialization.

**Join order:** hash join only in v1 (inner, left, semi, anti via mark-join flag column —
same trick as GQE's `GQE_JOIN_USE_MARK_JOIN`); nested-loop and range joins fall back.

**Pipelining:** per-partition tasks queued on 2–3 MLX streams:
`decode(p) ∥ compute(p-1) ∥ schedule(p+1)`. This is GQE Figure 3 minus the H2D stage.
DuckDB's own task scheduler drives the CPU side; one dedicated thread owns Metal
submission to avoid queue contention.

**Result hand-back:** GPU result partitions are already in unified memory; wrap as
DuckDB Vectors where layout permits, otherwise one memcpy into a DataChunk. TOP-N /
aggregates are tiny; large materializations stream chunk-by-chunk.

---

## 4. Operator & Type Coverage Matrix (v1 targets)

| | INT32/64 | FLOAT/DOUBLE | DECIMAL(≤18) | DATE/TIMESTAMP | VARCHAR (dict) |
|---|---|---|---|---|---|
| Filter / Projection | P1 | P1 | P2 (as int64 + scale) | P1 (epoch ints) | P2 (eq/in only) |
| Ungrouped agg | P1 | P1 | P2 | P1 (min/max) | — |
| GROUP BY hash agg | P2 | P2 | P2 | P2 | P2 (on codes) |
| Hash JOIN (inner/semi/anti/left) | P2 | P3 | P3 | P2 | P3 (on codes) |
| ORDER BY / TOP-N | P2 | P2 | P3 | P2 | P3 |
| LIMIT / CTE passthrough | P1 | P1 | P1 | P1 | P1 |

(P1/P2/P3 = phase in the roadmap below.) Everything else — WINDOW, nested types,
LIST/STRUCT, regex, LIKE beyond prefix — declines to CPU. DECIMAL wider than 64-bit,
HUGEINT: fallback (no 128-bit GPU arithmetic in v1).

---

## 5. Repo & Build

```
duckdb-mlx/
├── CMakeLists.txt              # duckdb/extension-template based (like Sirius)
├── extension_config.cmake
├── duckdb/                     # submodule, pinned
├── third_party/
│   └── mlx/                    # submodule (mlx C++ core) or fetch; evaluate mlx-c
├── src/
│   ├── mlx_extension.cpp       # load/register, settings, optimizer hook
│   ├── planner/                # logical-plan walker, supported-subtree detection
│   ├── format/                 # MlxTableFormat: ingest, encode, zone maps, cache
│   ├── ops/                    # Tier A graph builders (project/filter/agg/sort)
│   ├── kernels/                # Tier B .metal sources + launch wrappers
│   │   ├── hash_join.metal
│   │   ├── groupby.metal
│   │   ├── cascaded.metal
│   │   └── compact.metal
│   ├── exec/                   # pipeline scheduler, streams, fallback runtime
│   └── bridge/                 # DataChunk ⇄ mlx::array conversion
├── test/
│   ├── sql/                    # sqllogictest files (DuckDB harness)
│   └── cpp/                    # kernel unit tests (catch2)
├── benchmark/                  # TPC-H harness, GPU-vs-CPU diff runner
└── docs/
```

- Language: C++17 (DuckDB requirement) linking `mlx::core`; Metal kernels precompiled
  to a `.metallib` embedded in the extension binary.
- Platform gate: macOS 14+ (baseline), feature-flag paths for macOS 15
  (`MTLResidencySet`) and Apple-family-9 GPUs (M3/M4: 64-bit atomics, if adopted).
- Toolchain constraint to verify early: DuckDB community extensions build with a
  specific toolchain; Metal compilation needs Xcode CLT — CI on `macos-14/15` GitHub
  Actions arm64 runners.
- License: MIT (matches DuckDB + MLX); Sirius is Apache-2.0 — concepts are fine to
  port, avoid verbatim code copying or preserve headers if any is adapted.

---

## 6. Phased Roadmap

### Phase 0 — Feasibility spikes (1–2 weeks, throwaway code)
Kill risks before committing to architecture. Each spike has a go/no-go output.
1. **MLX-in-extension:** build a trivial DuckDB extension that links mlx::core, creates
   an array, evals a reduction. Verify no symbol/runtime conflicts (exceptions, RTTI,
   allocator interplay, extension ABI).
2. **Zero-copy bridge:** DuckDB flat Vector → `mx::array` without copy; measure
   copy-vs-nocopy for 1M/16M rows. Decide default ingestion path.
3. **Hash table kernel:** standalone MSL open-addressing build/probe at 10M/100M keys;
   measure vs DuckDB CPU join on the same machine. This number decides whether the
   whole project is worth it — target ≥3× on M-series Max for the raw primitive.
4. **Cascaded decode kernel:** decode throughput target ≥ 100 GB/s effective on M-Max
   class (should be easy; it's a few ALU ops per element).
5. **Threading:** MLX stream submission from DuckDB pipeline threads — contention test.

### Phase 1 — Skeleton & first transparent query (2–3 weeks)
- Extension template, settings, logging (spdlog, mirroring Sirius's `SET` knobs).
- Optimizer hook + plan walker; translate `SCAN → FILTER → PROJ → UNGROUPED AGG` on
  numeric columns; everything else falls back.
- Cold-path ingestion (densify to plain uncompressed partitions), no compression yet.
- sqllogictest suite: every supported query also runs with `mlx_execution=false` and
  results diffed automatically. **This differential harness is permanent infrastructure.**
- *Milestone:* TPC-H Q6 (SF1–SF10) runs on GPU end-to-end, correct, and ≥ parity.

### Phase 2 — Group-by, sort, pruning, compression (4–6 weeks)
- `groupby_hash` kernel + composite-key ORDER BY/TOP-N.
- Zone maps + pruning (reuse DuckDB row-group stats at bind, GPU zone maps for cache).
- Cascaded encoding + fused decode in the scan pipeline; GQE-style dual-heuristic
  algorithm selection (plain vs cascaded).
- Partition pipelining across streams.
- *Milestone:* TPC-H Q1 competitive; Q6-with-selective-predicates shows pruning win;
  cache-resident re-runs show ≥2× vs DuckDB hot CPU.

### Phase 3 — Joins & strings (6–8 weeks; the hard one)
- `hash_build/probe` with mark-join for semi/anti; radix partitioning fallback for
  large builds; join-order comes from DuckDB's optimizer (we translate, not re-plan).
- Dictionary string columns end-to-end (encode at ingest, join/group on codes, late
  materialize).
- Runtime fallback: OOM/failed kernel → transparent CPU re-execution.
- *Milestone:* ≥ 15 of 22 TPC-H queries fully on GPU at SF10–SF100; aggregate speedup
  reported honestly per query class (the Sirius/GQE comparison chart, Apple edition).

### Phase 4 — Hardening & release (4 weeks)
- Memory budget/eviction, `recommendedMaxWorkingSetSize` respect, pressure handling.
- Concurrency: multiple connections, serialized GPU queue with fair scheduling.
- Docs, `mlx_stats()` table function (cache contents, prune rates, kernel timings).
- Community-extension packaging (`INSTALL duckdb_mlx FROM community`).
- Blog post + benchmark reproduction scripts (Haybarn distribution angle: ship it as a
  bundled extension there first, where we control the toolchain).

### Phase 5+ — Stretch
- Hybrid CPU/GPU plans (GPU subtree bridge operator).
- Parquet-direct GPU decode (PLAIN/RLE/dictionary pages) — the io_uring-reader analog.
- WINDOW functions, DECIMAL128 via two-limb arithmetic, ANE experiments for filters.
- Optional Substrait consumer for GQE-comparable standalone benchmarking.

---

## 7. Benchmarking Plan
- **Hardware matrix:** M2 Pro (bandwidth-poor floor), M3/M4 Max (target), M4 Ultra if
  available. Report GB/s-normalized numbers so results transfer.
- **Workloads:** TPC-H SF1/SF10/SF100 (SF100 ≈ 30 GB compressed — fits unified memory
  on 64–128 GB machines, mirroring GQE's "data in CPU memory" setup one level down);
  ClickBench subset for wide-scan realism.
- **Baselines:** same-machine DuckDB (all cores), and *always hot-cache averaged over
  5 runs* to match GQE methodology; report GPU cold (ingest included) separately.
- **Per-query attribution:** prune rate, bytes decoded, kernel time vs graph-eval
  overhead — so we know *why* each query wins or loses, GQE-style.
- Include the standard TPC disclaimer (non-audited, non-comparable).

---

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Shared bandwidth ⇒ scan-bound queries show no win | High | Set expectations up front; lead with join/agg/cache-residency wins; compression recovers effective bandwidth |
| MLX abstraction overhead (lazy graph eval, allocator) for DB-style kernels | Medium | Tier B kernels bypass MLX graphs; MLX used where fusion pays. Escape hatch: talk raw Metal + keep MLX only as allocator/interop, decided by Phase 0 numbers |
| 32-bit-atomic hash tables limit key widths | Medium | Tag+verify design; 64-bit path on M3+; radix partitioning caps table sizes |
| DuckDB internal APIs churn (optimizer hook, Vector internals) | Medium | Pin DuckDB submodule per release; Sirius proves the hook surface is maintainable |
| macOS GPU watchdog kills long kernels | Medium | Partition-sized kernels (bounded work per dispatch) — the pipelining design already enforces this |
| String-heavy workloads dominated by dictionary build cost | Medium | Amortize at ingest; fall back when dictionary ratio poor |
| MLX license/version drift, mlx-c vs C++ API instability | Low | Pin submodule; wrap in thin internal interface (`bridge/`) |
| Correctness divergence (float reduction order, NULL semantics) | High-impact | Differential test harness from Phase 1; Kahan/pairwise sums for DOUBLE aggs; NULL-mask semantics tested exhaustively |

---

## 9. Open Questions (to resolve during Phase 0)
1. MLX C++ core directly vs `mlx-c`? (C++ likely, but check exception/ABI friction
   inside DuckDB extension loading.)
2. Consume DuckDB **logical** plan (more stable, we do physical decisions) vs
   **physical** plan (closer to execution, more churn)? Sirius intercepts at optimizer
   level → logical. Default: logical.
3. Partition size: 1M rows vs matching DuckDB's 122,880-row row group (GQE default was
   10M at 1 TB scale — ours should be smaller; tune in Phase 2).
4. Should the GPU cache persist across connections (process-lifetime singleton, like
   GQE's task-manager memory pool split from query memory)? Default: yes, with the
   two-pool split GQE uses (`query memory` vs `table/cache memory`).
5. Encode Decimal as scaled int64 always, or fall back when scale mismatch in joins?
6. Do we want a `gpu_processing`-style explicit API (`mlx_query('...')`) in addition to
   transparent interception, as Sirius offers both? Cheap to add, useful for debugging.

---

## 10. Concept Traceability (GQE/Sirius → duckdb-mlx)

| Winner concept | Source | Ported as |
|---|---|---|
| Optimizer-hook transparent interception + silent CPU fallback | Sirius | §3.1 Query layer |
| Extension-template packaging inside DuckDB | Sirius | §5 repo layout |
| Row-group/partition in-memory table format hiding compression & pruning from executor | GQE | §3.2 MlxTableFormat |
| Hybrid per-column compression w/ ratio heuristics (Cascaded vs LZ4) | GQE | Cascaded vs plain (LZ4 dropped — no DE hardware) |
| Zone-map filter pruning pre-execution | GQE | Two-tier pruning (DuckDB stats + GPU zone maps) |
| Pipelined stage overlap across row groups / streams | GQE | 2-stage decode∥compute pipeline (H2D stage eliminated) |
| Batched transfers (cudaMemcpyBatchAsync) | GQE | Obsolete under unified memory → batched *layout conversion* on ingest |
| Perfect-hash join/groupby, mark join for semi/anti | GQE env flags | Tier B kernels, mark-join in Phase 3 |
| Query-memory vs task-manager-memory pool split | GQE | §9 Q4, cache vs query pools |
| Hot-cache 5-run TPC-H methodology + honest disclaimer | GQE | §7 benchmarking |
