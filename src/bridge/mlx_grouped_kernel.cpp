#include "mlx_groupby_detail.hpp"

#include <mlx/fast.h>
#include <mlx/mlx.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace duckdb_mlx {

namespace {

constexpr int kMaxTileValProgs = 8;
constexpr int kMaxTileSlots = 512;
constexpr int kMaxTileCard = 64;
constexpr int kTileRows = 131072;
constexpr int kThreadsPerGroup = 256;

//! Shared Metal helpers: parallel tree reduce within a threadgroup.
const char *TileKernelHeader() {
	return R"(
            template <typename T>
            void tg_tree_add_n(threadgroup T *buf, int lid, int n) {
                for (int off = n / 2; off > 0; off >>= 1) {
                    if (lid < off) {
                        buf[lid] += buf[lid + off];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        )";
}

const mx::fast::CustomKernelFunction &GroupedTileAccumulateKernel() {
	static auto kernel = []() {
		const std::string src = R"(
            const int tile = 131072;
            const int tgid = int(threadgroup_position_in_grid.x);
            const int start = tgid * tile;
            const int nrows = int(codes_shape[0]);
            const int end = min(nrows, start + tile);
            const int lid = int(thread_position_in_threadgroup.x);

            long local_sum[SLOTS];
            int local_cnt[SLOTS];
            int local_rows[CARD];
            for (int i = 0; i < SLOTS; ++i) {
                local_sum[i] = 0;
                local_cnt[i] = 0;
            }
            for (int i = 0; i < CARD; ++i) {
                local_rows[i] = 0;
            }

            for (int row = start + lid; row < end; row += 256) {
                if (pass[row] == 0) {
                    continue;
                }
                int g = codes[row];
                if (g < 0 || g >= CARD) {
                    continue;
                }
                local_rows[g] += 1;
                for (int p = 0; p < VAL_N; ++p) {
                    uchar vm = 1;
                    switch (p) {
                    case 0: vm = vmask0[row]; break;
                    case 1: vm = vmask1[row]; break;
                    case 2: vm = vmask2[row]; break;
                    case 3: vm = vmask3[row]; break;
                    case 4: vm = vmask4[row]; break;
                    case 5: vm = vmask5[row]; break;
                    case 6: vm = vmask6[row]; break;
                    case 7: vm = vmask7[row]; break;
                    default: break;
                    }
                    if (vm == 0) {
                        continue;
                    }
                    long v = 0;
                    switch (p) {
                    case 0: v = val0[row]; break;
                    case 1: v = val1[row]; break;
                    case 2: v = val2[row]; break;
                    case 3: v = val3[row]; break;
                    case 4: v = val4[row]; break;
                    case 5: v = val5[row]; break;
                    case 6: v = val6[row]; break;
                    case 7: v = val7[row]; break;
                    default: break;
                    }
                    int slot = g * VAL_N + p;
                    local_sum[slot] += v;
                    local_cnt[slot] += 1;
                }
            }

            const int REDUCE_BATCH = 4;
            threadgroup long tg_sum[REDUCE_BATCH * 256];
            threadgroup int tg_cnt[REDUCE_BATCH * 256];
            threadgroup int tg_rows[REDUCE_BATCH * 256];

            threadgroup long block_sum[SLOTS];
            threadgroup int block_cnt[SLOTS];
            threadgroup int block_rows[CARD];

            for (int base = 0; base < SLOTS; base += REDUCE_BATCH) {
                int batch_n = min(REDUCE_BATCH, SLOTS - base);
                for (int i = 0; i < batch_n; ++i) {
                    int s = base + i;
                    tg_sum[i * 256 + lid] = local_sum[s];
                    tg_cnt[i * 256 + lid] = local_cnt[s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int off = 128; off > 0; off >>= 1) {
                    for (int i = 0; i < batch_n; ++i) {
                        if (lid < off) {
                            tg_sum[i * 256 + lid] += tg_sum[i * 256 + lid + off];
                            tg_cnt[i * 256 + lid] += tg_cnt[i * 256 + lid + off];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                for (int i = 0; i < batch_n; ++i) {
                    if (lid == 0) {
                        int s = base + i;
                        block_sum[s] = tg_sum[i * 256 + 0];
                        block_cnt[s] = tg_cnt[i * 256 + 0];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            for (int base = 0; base < CARD; base += REDUCE_BATCH) {
                int batch_n = min(REDUCE_BATCH, CARD - base);
                for (int i = 0; i < batch_n; ++i) {
                    int g = base + i;
                    tg_rows[i * 256 + lid] = local_rows[g];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int off = 128; off > 0; off >>= 1) {
                    for (int i = 0; i < batch_n; ++i) {
                        if (lid < off) {
                            tg_rows[i * 256 + lid] += tg_rows[i * 256 + lid + off];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                for (int i = 0; i < batch_n; ++i) {
                    if (lid == 0) {
                        int g = base + i;
                        block_rows[g] = tg_rows[i * 256 + 0];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lid == 0) {
                for (int s = 0; s < SLOTS; ++s) {
                    partial_sums[tgid * SLOTS + s] = block_sum[s];
                    partial_counts[tgid * SLOTS + s] = block_cnt[s];
                }
                for (int g = 0; g < CARD; ++g) {
                    partial_rows[tgid * CARD + g] = block_rows[g];
                }
            }
        )";
		return mx::fast::metal_kernel("mlx_grouped_tile_i64",
		                              {"codes", "pass", "val0", "val1", "val2", "val3", "val4", "val5", "val6",
		                               "val7", "vmask0", "vmask1", "vmask2", "vmask3", "vmask4", "vmask5", "vmask6",
		                               "vmask7"},
		                              {"partial_rows", "partial_sums", "partial_counts"}, src, TileKernelHeader(), true,
		                              false);
	}();
	return kernel;
}

const mx::fast::CustomKernelFunction &GroupedPackedTileKernel() {
	static auto kernel = []() {
		const std::string src = R"(
            const int tile = 131072;
            const int tgid = int(threadgroup_position_in_grid.x);
            const int start = tgid * tile;
            const int nrows = int(codes_shape[0]);
            const int end = min(nrows, start + tile);
            const int lid = int(thread_position_in_threadgroup.x);
            const int pack_n = int(packed_shape[1]);

            long local_sum[SLOTS];
            int local_cnt[SLOTS];
            int local_rows[CARD];
            for (int i = 0; i < SLOTS; ++i) {
                local_sum[i] = 0;
                local_cnt[i] = 0;
            }
            for (int i = 0; i < CARD; ++i) {
                local_rows[i] = 0;
            }

            for (int row = start + lid; row < end; row += 256) {
                if (pass[row] == 0) {
                    continue;
                }
                int g = codes[row];
                if (g < 0 || g >= CARD) {
                    continue;
                }
                local_rows[g] += 1;
                const int row_base = row * pack_n;
                for (int p = 0; p < VAL_N; ++p) {
                    int pc = col_map[p];
                    if (pc < 0 || pc >= pack_n) {
                        continue;
                    }
                    long v = packed[row_base + pc];
                    int slot = g * VAL_N + p;
                    local_sum[slot] += v;
                    local_cnt[slot] += 1;
                }
            }

            const int REDUCE_BATCH = 4;
            threadgroup long tg_sum[REDUCE_BATCH * 256];
            threadgroup int tg_cnt[REDUCE_BATCH * 256];
            threadgroup int tg_rows[REDUCE_BATCH * 256];

            threadgroup long block_sum[SLOTS];
            threadgroup int block_cnt[SLOTS];
            threadgroup int block_rows[CARD];

            for (int base = 0; base < SLOTS; base += REDUCE_BATCH) {
                int batch_n = min(REDUCE_BATCH, SLOTS - base);
                for (int i = 0; i < batch_n; ++i) {
                    int s = base + i;
                    tg_sum[i * 256 + lid] = local_sum[s];
                    tg_cnt[i * 256 + lid] = local_cnt[s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int off = 128; off > 0; off >>= 1) {
                    for (int i = 0; i < batch_n; ++i) {
                        if (lid < off) {
                            tg_sum[i * 256 + lid] += tg_sum[i * 256 + lid + off];
                            tg_cnt[i * 256 + lid] += tg_cnt[i * 256 + lid + off];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                for (int i = 0; i < batch_n; ++i) {
                    if (lid == 0) {
                        int s = base + i;
                        block_sum[s] = tg_sum[i * 256 + 0];
                        block_cnt[s] = tg_cnt[i * 256 + 0];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            for (int base = 0; base < CARD; base += REDUCE_BATCH) {
                int batch_n = min(REDUCE_BATCH, CARD - base);
                for (int i = 0; i < batch_n; ++i) {
                    int g = base + i;
                    tg_rows[i * 256 + lid] = local_rows[g];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int off = 128; off > 0; off >>= 1) {
                    for (int i = 0; i < batch_n; ++i) {
                        if (lid < off) {
                            tg_rows[i * 256 + lid] += tg_rows[i * 256 + lid + off];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                for (int i = 0; i < batch_n; ++i) {
                    if (lid == 0) {
                        int g = base + i;
                        block_rows[g] = tg_rows[i * 256 + 0];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lid == 0) {
                for (int s = 0; s < SLOTS; ++s) {
                    partial_sums[tgid * SLOTS + s] = block_sum[s];
                    partial_counts[tgid * SLOTS + s] = block_cnt[s];
                }
                for (int g = 0; g < CARD; ++g) {
                    partial_rows[tgid * CARD + g] = block_rows[g];
                }
            }
        )";
		return mx::fast::metal_kernel("mlx_grouped_packed_tile", {"codes", "pass", "packed", "col_map"},
		                              {"partial_rows", "partial_sums", "partial_counts"}, src, TileKernelHeader(), true,
		                              false);
	}();
	return kernel;
}

const mx::fast::CustomKernelFunction &GroupedTileMergeKernel() {
	static auto kernel = []() {
		const std::string src = R"(
            const int tg = int(threadgroup_position_in_grid.x);
            const int lid = int(thread_position_in_threadgroup.x);
            const int ntiles = int(partial_sums_shape[0]);

            threadgroup long tg_sum[256];
            threadgroup int tg_cnt[256];
            threadgroup int tg_rows[256];

            if (tg < SLOTS) {
                long acc_sum = 0;
                int acc_cnt = 0;
                for (int t = lid; t < ntiles; t += 256) {
                    acc_sum += partial_sums[t * SLOTS + tg];
                    acc_cnt += partial_counts[t * SLOTS + tg];
                }
                tg_sum[lid] = acc_sum;
                tg_cnt[lid] = acc_cnt;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                tg_tree_add_n(tg_sum, lid, 256);
                tg_tree_add_n(tg_cnt, lid, 256);
                if (lid == 0) {
                    final_sums[tg] = tg_sum[0];
                    final_counts[tg] = tg_cnt[0];
                }
            }

            if (tg < CARD) {
                int acc_rows = 0;
                for (int t = lid; t < ntiles; t += 256) {
                    acc_rows += partial_rows[t * CARD + tg];
                }
                tg_rows[lid] = acc_rows;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                tg_tree_add_n(tg_rows, lid, 256);
                if (lid == 0) {
                    final_rows[tg] = tg_rows[0];
                }
            }
        )";
		return mx::fast::metal_kernel("mlx_grouped_tile_merge", {"partial_rows", "partial_sums", "partial_counts"},
		                              {"final_rows", "final_sums", "final_counts"}, src, TileKernelHeader(), true, false);
	}();
	return kernel;
}

std::vector<std::pair<std::string, mx::fast::TemplateArg>> TileTemplateArgs(int card, int val_n) {
	const int slots = card * val_n;
	return {{"CARD", card}, {"VAL_N", val_n}, {"SLOTS", slots}};
}

bool ProgramsUseTileKernel(const std::vector<MlxSumProgram> &programs, int64_t card) {
	if (card <= 0 || card > kMaxTileCard) {
		return false;
	}
	size_t val_progs = 0;
	for (auto &program : programs) {
		switch (program.kind) {
		case MlxAggKind::COUNT_STAR:
			break;
		case MlxAggKind::SUM:
		case MlxAggKind::AVG:
			if (!program.int_lane) {
				return false;
			}
			val_progs++;
			break;
		case MlxAggKind::COUNT:
			break;
		case MlxAggKind::MIN:
		case MlxAggKind::MAX:
			return false;
		default:
			return false;
		}
	}
	return val_progs > 0 && static_cast<int64_t>(val_progs * card) <= kMaxTileSlots;
}

void MergeTilePartials(MlxGroupedState &state, int card, int val_n, size_t nprogs,
                       const std::vector<std::pair<size_t, int>> &val_slots, const std::vector<MlxSumProgram> &programs,
                       mx::array rows_reduced, mx::array sums_reduced, mx::array counts_reduced) {
	auto rows_ptr = rows_reduced.data<int32_t>();
	auto sums_ptr = sums_reduced.data<int64_t>();
	auto counts_ptr = counts_reduced.data<int32_t>();

	for (int g = 0; g < card; g++) {
		state.rows[static_cast<size_t>(g)] += rows_ptr[g];
	}
	for (int vp = 0; vp < val_n; vp++) {
		auto p = val_slots[static_cast<size_t>(vp)].first;
		auto kind = programs[p].kind;
		for (int g = 0; g < card; g++) {
			auto slot = static_cast<size_t>(g) * nprogs + p;
			auto kslot = g * val_n + vp;
			if (kind == MlxAggKind::SUM || kind == MlxAggKind::AVG) {
				state.ivalues[slot] += static_cast<__int128>(sums_ptr[kslot]);
				state.counts[slot] += counts_ptr[kslot];
			}
		}
	}
}

void RunTileAccumulateAndMerge(MlxGroupedState &state, int card, int val_n, size_t nprogs, int n, int tiles,
                               mx::array partial_rows, mx::array partial_sums, mx::array partial_counts,
                               const std::vector<std::pair<size_t, int>> &val_slots,
                               const std::vector<MlxSumProgram> &programs) {
	const int slots = card * val_n;
	const int merge_groups = std::max(slots, card);
	const int merge_grid = merge_groups * kThreadsPerGroup;
	auto template_args = TileTemplateArgs(card, val_n);

	auto merged = GroupedTileMergeKernel()(
	    {partial_rows, partial_sums, partial_counts},
	    {mx::Shape {card}, mx::Shape {slots}, mx::Shape {slots}}, {mx::int32, mx::int64, mx::int32},
	    std::make_tuple(merge_grid, 1, 1), std::make_tuple(kThreadsPerGroup, 1, 1), template_args, 0.0f, false,
	    mx::Device::gpu);

	mx::eval({merged[0], merged[1], merged[2]});
	MergeTilePartials(state, card, val_n, nprogs, val_slots, programs, merged[0], merged[1], merged[2]);
}

} // namespace

bool GroupedTileKernelEligible(const std::vector<MlxSumProgram> &programs, int64_t card) {
	return ProgramsUseTileKernel(programs, card);
}

void GroupedTileKernelAccumulate(MlxGroupedState &state, const mx::array &codes,
                                 const std::vector<std::pair<size_t, mx::array>> &val_exprs,
                                 const std::vector<std::optional<mx::array>> &val_masks,
                                 const std::vector<MlxSumProgram> &programs) {
	if (codes.shape(0) == 0 || !ProgramsUseTileKernel(programs, state.card)) {
		throw std::runtime_error("GroupedTileKernelAccumulate: ineligible program/card shape");
	}
	auto card = static_cast<int>(state.card);
	auto n = static_cast<int>(codes.shape(0));
	auto nprogs = programs.size();
	const int val_n = static_cast<int>(val_exprs.size());
	const int slots = card * val_n;
	if (val_n == 0 || val_n > kMaxTileValProgs || slots > kMaxTileSlots) {
		throw std::runtime_error("GroupedTileKernelAccumulate: val program layout unsupported");
	}

	auto dump = mx::array(card, mx::int32);
	auto zero_i32 = mx::array(0, mx::int32);
	auto in_range = mx::logical_and(mx::greater_equal(codes, zero_i32), mx::less(codes, dump));
	auto pass = mx::astype(in_range, mx::uint8);

	std::vector<mx::array> val_inputs;
	std::vector<mx::array> mask_inputs;
	val_inputs.reserve(kMaxTileValProgs);
	mask_inputs.reserve(kMaxTileValProgs);
	auto ones = mx::ones({n}, mx::uint8);
	for (int i = 0; i < kMaxTileValProgs; i++) {
		if (i < val_n) {
			val_inputs.push_back(val_exprs[static_cast<size_t>(i)].second);
			mask_inputs.push_back((i < static_cast<int>(val_masks.size()) && val_masks[static_cast<size_t>(i)].has_value())
			                          ? *val_masks[static_cast<size_t>(i)]
			                          : ones);
		} else {
			val_inputs.push_back(mx::zeros({n}, mx::int64));
			mask_inputs.push_back(ones);
		}
	}

	int tiles = (n + kTileRows - 1) / kTileRows;
	int grid_threads = std::max(tiles, 1) * kThreadsPerGroup;
	auto template_args = TileTemplateArgs(card, val_n);

	auto built = GroupedTileAccumulateKernel()(
	    {codes, pass, val_inputs[0], val_inputs[1], val_inputs[2], val_inputs[3], val_inputs[4], val_inputs[5],
	     val_inputs[6], val_inputs[7], mask_inputs[0], mask_inputs[1], mask_inputs[2], mask_inputs[3], mask_inputs[4],
	     mask_inputs[5], mask_inputs[6], mask_inputs[7]},
	    {mx::Shape {tiles, card}, mx::Shape {tiles, slots}, mx::Shape {tiles, slots}},
	    {mx::int32, mx::int64, mx::int32}, std::make_tuple(grid_threads, 1, 1), std::make_tuple(kThreadsPerGroup, 1, 1),
	    template_args, 0.0f, false, mx::Device::gpu);

	std::vector<std::pair<size_t, int>> val_slots;
	val_slots.reserve(val_exprs.size());
	for (size_t i = 0; i < val_exprs.size(); i++) {
		val_slots.push_back({val_exprs[i].first, static_cast<int>(i)});
	}
	RunTileAccumulateAndMerge(state, card, val_n, nprogs, n, tiles, built[0], built[1], built[2], val_slots, programs);
}

void GroupedPackedTileAccumulate(MlxGroupedState &state, const mx::array &codes, const mx::array &pass,
                                 const mx::array &packed, int val_n,
                                 const std::vector<std::pair<size_t, int>> &val_slots,
                                 const std::vector<MlxSumProgram> &programs) {
	if (codes.shape(0) == 0 || !ProgramsUseTileKernel(programs, state.card)) {
		throw std::runtime_error("GroupedPackedTileAccumulate: ineligible program/card shape");
	}
	auto card = static_cast<int>(state.card);
	auto n = static_cast<int>(codes.shape(0));
	auto nprogs = programs.size();
	const int slots = card * val_n;
	if (val_n <= 0 || val_n > kMaxTileValProgs || slots > kMaxTileSlots) {
		throw std::runtime_error("GroupedPackedTileAccumulate: val program layout unsupported");
	}

	auto dump = mx::array(card, mx::int32);
	auto zero_i32 = mx::array(0, mx::int32);
	auto in_range = mx::logical_and(mx::greater_equal(codes, zero_i32), mx::less(codes, dump));
	auto combined_pass = mx::logical_and(mx::astype(pass, mx::bool_), in_range);
	auto pass_u8 = mx::astype(combined_pass, mx::uint8);

	std::vector<int32_t> col_map(static_cast<size_t>(val_n), 0);
	for (int i = 0; i < val_n; i++) {
		col_map[static_cast<size_t>(i)] = val_slots[static_cast<size_t>(i)].second;
	}
	mx::array col_map_arr(col_map.data(), mx::Shape {val_n}, mx::int32);

	int tiles = (n + kTileRows - 1) / kTileRows;
	int grid_threads = std::max(tiles, 1) * kThreadsPerGroup;
	auto template_args = TileTemplateArgs(card, val_n);

	auto built = GroupedPackedTileKernel()(
	    {codes, pass_u8, packed, col_map_arr}, {mx::Shape {tiles, card}, mx::Shape {tiles, slots}, mx::Shape {tiles, slots}},
	    {mx::int32, mx::int64, mx::int32}, std::make_tuple(grid_threads, 1, 1), std::make_tuple(kThreadsPerGroup, 1, 1),
	    template_args, 0.0f, false, mx::Device::gpu);

	RunTileAccumulateAndMerge(state, card, val_n, nprogs, n, tiles, built[0], built[1], built[2], val_slots, programs);
}

} // namespace duckdb_mlx
