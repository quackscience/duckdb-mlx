#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace duckdb_mlx {

// Thin interface over mlx::core, same isolation pattern as mlx_logger: DuckDB
// translation units never include MLX headers, MLX translation units never
// include DuckDB headers. Data crosses as raw pointers into unified memory.

std::string MlxVersion();

//! Runs a small end-to-end computation on the default (GPU) device.
//! Returns "ok" on success, otherwise a description of the failure.
std::string MlxSelftest();

//! Sums `count` int64 values on the GPU.
int64_t MlxSumInt64(const int64_t *data, size_t count);

//! Benchmark helper: ALU-dense fused expression sum(sin(x)*cos(x)+sqrt(|x|+1))
//! in float32 on the GPU. Timing-comparable to the CPU expression, not
//! bit-comparable (fp32 vs fp64).
double MlxExprBenchInt64(const int64_t *data, size_t count);

// GPU-resident vector similarity search (the "MlxTableFormat" cache concept,
// specialized to embedding matrices). Pinned matrices are process-lifetime
// and L2-normalized once at pin time, so search is one matvec + top-k.

//! Copies an N x dim fp32 row-major matrix to a named GPU-resident,
//! L2-normalized mx::array, stored as fp16 when `half` (halves bandwidth,
//! standard for embeddings). Replaces any existing matrix of that name.
//! Returns the number of pinned rows.
int64_t MlxVssPin(const std::string &name, const float *data, int64_t n, int64_t dim, bool half);

struct MlxVssMatch {
	int64_t index;
	float score;
};

//! Cosine top-k of `query` (length dim) against the pinned matrix `name`.
//! Throws std::runtime_error if the name is unknown or dim mismatches.
std::vector<MlxVssMatch> MlxVssSearch(const std::string &name, const float *query, int64_t dim, int64_t k);

struct MlxVssBatchMatch {
	int64_t query;
	int64_t index;
	float score;
};

//! Cosine top-k of `q` row-major (Q x dim) queries against the pinned matrix
//! in a single matmul. Returns Q*k matches ordered by (query, rank).
std::vector<MlxVssBatchMatch> MlxVssSearchBatch(const std::string &name, const float *queries, int64_t q, int64_t dim,
                                                int64_t k);

// Transparent execution (PLAN Phase 1): the planner translates supported
// DuckDB expression trees into this postfix IR; the bridge evaluates the
// whole program as one fused MLX graph over column buffers.

enum class MlxExprOpCode : uint8_t {
	LOAD_COL,  // push column `col`
	CONST_VAL, // push constant `value` (or `ivalue` when int_lane)
	ADD,
	SUB,
	MUL,
	DIV,
	NEGATE,
	SIN,
	COS,
	SQRT,
	ABS,
	//! int64 -> float32 followed by multiply with `value` (decimal -> double
	//! casts use 1/10^scale)
	TO_FLOAT,
	// predicates (produce boolean arrays)
	CMP_LT,
	CMP_LE,
	CMP_GT,
	CMP_GE,
	CMP_EQ,
	CMP_NE,
	AND,
	OR,
	NOT,
};

//! Two evaluation lanes: fp32 (numeric/date columns) and exact int64
//! (DECIMAL columns as raw scaled integers — fp32 would silently corrupt
//! them). Lanes only meet at comparisons (same-lane operands enforced by the
//! translator) and TO_FLOAT.
struct MlxExprOp {
	MlxExprOpCode code;
	int32_t col = 0;
	double value = 0;
	int64_t ivalue = 0;    // integer-lane constant (raw decimal, exceeds 2^53)
	bool int_lane = false; // lane of a CONST_VAL push
};

enum class MlxAggKind : uint8_t {
	SUM,
	COUNT,
	COUNT_STAR,
	AVG,
	MIN,
	MAX,
};

struct MlxSumProgram {
	MlxAggKind kind = MlxAggKind::SUM;
	//! Value expression in postfix form; empty for COUNT_STAR
	std::vector<MlxExprOp> ops;
	//! Columns whose NULL mask excludes a row from this aggregate
	std::vector<int32_t> null_cols;
	//! Exact int64 evaluation (DECIMAL expressions); results land in ivalue
	bool int_lane = false;
	//! Multiplier the planner applies when rendering a DOUBLE result from a
	//! raw-scaled int-lane value (e.g. avg(DECIMAL) => 10^-scale)
	double render_scale = 1.0;
};

//! WHERE-clause predicate applied to every aggregate (rows where it is false
//! or NULL are excluded). Empty ops = no filter.
struct MlxFilter {
	std::vector<MlxExprOp> ops;
	//! Columns whose NULL value makes the predicate non-true for that row
	std::vector<int32_t> null_cols;
};

struct MlxColumnData {
	const float *values = nullptr;    // fp32 lane (nulls hold 0.0)
	const uint8_t *valid = nullptr;   // 1 = valid; nullptr = all valid
	const int64_t *ivalues = nullptr; // int64 lane (raw scaled decimals)
};

struct MlxSumResult {
	double value;
	int64_t valid_count; // 0 => SQL NULL (for COUNT kinds the result itself)
	int64_t ivalue = 0;  // exact result for int-lane programs
};

//! Evaluates each aggregate program over `count` rows of `cols` on the GPU
//! (fp32), honoring SQL NULL semantics and the shared WHERE filter.
std::vector<MlxSumResult> MlxSumExprs(const std::vector<MlxColumnData> &cols, size_t count,
                                      const std::vector<MlxSumProgram> &programs, const MlxFilter &filter);

// GPU-resident column cache — the GQE "in-memory table format" analog. The
// first intercepted query over a table populates it; subsequent queries are
// served entirely from GPU-resident columns with no table scan at all.
// Columns are keyed per storage id; missing columns in a new query are
// populated incrementally while reusing row-aligned segments already cached.

struct MlxZoneMap {
	double min_val = 0;
	double max_val = 0;
	//! exact bounds for int64-lane columns (raw decimals exceed 2^53)
	int64_t imin = 0;
	int64_t imax = 0;
	bool int_lane = false;
	bool has_null = false;
};

struct MlxCacheStats {
	int64_t segments_total = 0;
	int64_t segments_pruned = 0;
};

//! Drops every cached column whose key starts with `prefix`.
void MlxCacheDrop(const std::string &prefix);

//! True when all keys are cached, row counts equal expected_rows, and all
//! columns come from the same population (row-aligned).
bool MlxCacheHas(const std::vector<std::string> &keys, int64_t expected_rows);

//! Plan a cache population: returns which `col_keys` need new segments written.
//! Drops the whole table when `expected_rows` disagrees with a cached column.
//! Existing columns with matching rows are left intact (alternating subsets).
struct MlxPopulationPlan {
	int64_t population = 0;
	std::vector<bool> store_col;
};

MlxPopulationPlan MlxCacheBeginPopulation(const std::string &table_prefix, const std::vector<std::string> &col_keys,
                                          int64_t expected_rows);

//! Appends one row-aligned segment. Only columns with `store_col[i] == true` are
//! written; others are skipped (already resident).
void MlxCacheStoreSegment(int64_t population, const std::vector<std::string> &col_keys,
                          const std::vector<bool> &store_col, const std::vector<MlxColumnData> &cols, size_t count);

//! Evaluates aggregate programs over cached columns, entirely GPU-resident.
//! Partition-level zone maps prune segments before kernel launch when the WHERE
//! clause is a conjunction of column-vs-constant comparisons.
std::vector<MlxSumResult> MlxSumExprsCached(const std::vector<std::string> &col_keys,
                                            const std::vector<MlxSumProgram> &programs, const MlxFilter &filter);

//! Stats from the most recent cached evaluation (segment pruning).
MlxCacheStats MlxCacheLastStats();

//! Zone maps per segment for a cached column (empty when unknown).
std::vector<MlxZoneMap> MlxCacheColumnZoneMaps(const std::string &col_key);

//! Clears the GPU column cache (all tables). For tests.
void MlxCacheClearAll();

// GROUP BY hash / sort aggregation (Tier B + MLX scatter_add).

struct MlxGroupbyRow {
	int64_t key;
	double sum;
	int64_t count;
};

//! GPU group-by sum on int64 keys and fp32 values. `use_hash` selects the open-
//! addressing Metal kernel; otherwise uses argsort + scatter_add (fast for
//! moderate cardinality).
std::vector<MlxGroupbyRow> MlxGroupbySum(const int64_t *keys, const double *values, const uint8_t *valid, size_t count,
                                         bool use_hash = false);

//! Incremental perfect-hash table update while populating the GPU cache.
void MlxGroupbyDenseAccumulateHost(const std::string &group_col_key, const std::string &value_col_key,
                                   int64_t population, const float *group_values, const float *sum_values,
                                   size_t count);

//! Whether a cached GROUP BY over these columns is provably correct (dense
//! table ready, or fp32-exact integer keys and no NULLs per zone maps).
bool MlxGroupbyCachedSafe(const std::string &group_col_key, const std::string &value_col_key);

//! Group-by over GPU-resident cache columns; reads incremental dense table when available.
std::vector<MlxGroupbyRow> MlxGroupbySumCached(const std::string &group_col_key, const std::string &value_col_key);

//! Benchmark helper: returns total sum of per-group sums (checksum).
double MlxGroupbyBenchSum(const int64_t *keys, const double *values, size_t count, bool use_hash);

} // namespace duckdb_mlx
