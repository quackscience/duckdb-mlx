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

bool GroupedTileKernelEligible(const std::vector<MlxSumProgram> &programs, int64_t card);
void GroupedTileKernelAccumulate(MlxGroupedState &state, const mlx::core::array &codes,
                                 const std::vector<std::pair<size_t, mlx::core::array>> &val_exprs,
                                 const std::vector<std::optional<mlx::core::array>> &val_masks,
                                 const std::vector<MlxSumProgram> &programs);
void GroupedPackedTileAccumulate(MlxGroupedState &state, const mlx::core::array &codes, const mlx::core::array &pass,
                                 const mlx::core::array &packed, int val_n,
                                 const std::vector<std::pair<size_t, int>> &val_slots,
                                 const std::vector<MlxSumProgram> &programs);
void GroupedQ1FusedAccumulate(MlxGroupedState &state, const mlx::core::array &codes, const mlx::core::array &pass,
                              const mlx::core::array &packed, int val_n,
                              const std::vector<std::pair<size_t, int>> &val_slots,
                              const std::vector<MlxSumProgram> &programs);

struct MlxMultiAggResult {
	int64_t sum = 0;
	int64_t min = 0;
	int64_t max = 0;
	int64_t count = 0;
};

int64_t StreamingInt64Sum(const mlx::core::array &values);
MlxMultiAggResult StreamingMultiAgg(const mlx::core::array &values);

} // namespace duckdb_mlx
