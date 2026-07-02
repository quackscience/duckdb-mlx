#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

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

} // namespace duckdb_mlx
