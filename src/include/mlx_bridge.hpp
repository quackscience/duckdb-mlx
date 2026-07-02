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

} // namespace duckdb_mlx
