#pragma once

// Internal to duckdb_mlx_bridge — not included from DuckDB extension TUs.

#include "mlx_bridge.hpp"

#include <mlx/mlx.h>

namespace duckdb_mlx {

std::vector<MlxGroupbyRow> MlxGroupbySumArrays(mlx::core::array key_arr, mlx::core::array val_arr,
                                               bool use_hash = false, int64_t estimated_groups = -1);

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
//! Dedicated Q1 path: uint8 codes with the filter pre-folded (failing rows carry
//! code == card), deduplicated int32/int64 value lanes, counts derived from
//! per-group row counts (lanes must be NULL-free — enforced at pin time).
void GroupedQ1DedicatedAccumulate(MlxGroupedState &state, const mlx::core::array &codes8,
                                  const std::vector<mlx::core::array> &dist_cols,
                                  const std::vector<int32_t> &dist_of_pack,
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

//! gpudb v0.1.3 slot-lock hash aggregate (32K partitions × 1K TG slots).
std::vector<MlxGroupbyRow> GroupbySumSlotlockGpu(mlx::core::array keys, mlx::core::array vals);
bool GroupbySumSlotlockValid(mlx::core::array keys, mlx::core::array vals, const std::vector<MlxGroupbyRow> &rows);

//! gpudb LSD radix sort + host segment-reduce.
std::vector<MlxGroupbyRow> GroupbySumRadixGpu(mlx::core::array keys, mlx::core::array vals);
bool GroupbySumRadixValid(mlx::core::array keys, mlx::core::array vals, const std::vector<MlxGroupbyRow> &rows);

void MlxGroupbySetPathOverride(const char *path);
const char *MlxGroupbyPathOverride();
const char *GroupbyPathFromEnv();
bool GroupbyShouldTrySlotlock(int n, int64_t estimated_groups = -1);
bool GroupbyShouldTryRadix(int n, int64_t estimated_groups = -1);

} // namespace duckdb_mlx
