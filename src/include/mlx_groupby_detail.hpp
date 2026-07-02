#pragma once

// Internal to duckdb_mlx_bridge — not included from DuckDB extension TUs.

#include "mlx_bridge.hpp"

#include <mlx/mlx.h>

namespace duckdb_mlx {

std::vector<MlxGroupbyRow> MlxGroupbySumArrays(mlx::core::array key_arr, mlx::core::array val_arr,
                                               bool use_hash = false);

std::vector<MlxGroupbyRow> MlxGroupbySumDenseGpuArrays(mlx::core::array key_arr, mlx::core::array val_arr);

std::vector<MlxGroupbyRow> MlxGroupbyDenseTable(int64_t kmin, mlx::core::array sums, mlx::core::array counts);

void MlxGroupbyDenseAccumulate(const std::string &group_col_key, const std::string &value_col_key, int64_t population,
                               mlx::core::array key_arr, mlx::core::array val_arr);

bool MlxGroupbyDenseReady(const std::string &group_col_key, const std::string &value_col_key, int64_t population);

bool MlxGroupbyDenseTryRead(const std::string &group_col_key, const std::string &value_col_key, int64_t population,
                            std::vector<MlxGroupbyRow> &out);

void MlxGroupbyDenseClearTable(const std::string &table_prefix);

} // namespace duckdb_mlx
