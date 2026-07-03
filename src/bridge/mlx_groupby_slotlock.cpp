// Slot-lock GROUP BY SUM — ported from gpudb groupby.metal + metal_groupby.mm (v0.1.3).
// Apache-2.0 — https://github.com/singhpratech/duckdbgpumetaldbram

#include "mlx_groupby_detail.hpp"

#include <mlx/fast.h>
#include <mlx/mlx.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace duckdb_mlx {

namespace {

constexpr int kSlotPartitions = 32768;
constexpr int kSlotCount = 1024;
constexpr int kSlotTgThreads = 64;
constexpr int kCountThreads = 256;
constexpr int kCountWork = 256;

const char *SlotlockHashHeader() {
	return R"(
            inline ulong slotlock_hash(long key) {
                ulong x = (ulong)key;
                x ^= (x >> 33);
                x *= 0xff51afd7ed558ccdUL;
                x ^= (x >> 33);
                x *= 0xc4ceb9fe1a85ec53UL;
                x ^= (x >> 33);
                return x;
            }
            inline uint slotlock_partition(ulong h) {
                return (uint)(h & 32767u);
            }
            inline uint slotlock_slot_start(ulong h) {
                return (uint)((h >> 16) & 1023u);
            }
        )";
}

const mx::fast::CustomKernelFunction &PartitionCountKernel() {
	static auto kernel = []() {
		const std::string src = R"(
            uint gid = thread_position_in_grid.x;
            uint gsize = threads_per_grid.x;
            int nrows = int(keys_shape[0]);
            for (uint i = gid; i < uint(nrows); i += gsize) {
                const ulong h = slotlock_hash(keys[i]);
                const uint p = slotlock_partition(h);
                atomic_fetch_add_explicit(&partition_counts[p], 1u, memory_order_relaxed);
            }
        )";
		return mx::fast::metal_kernel("mlx_gb_partition_count", {"keys"}, {"partition_counts"}, src,
		                              SlotlockHashHeader(), true, false);
	}();
	return kernel;
}

const mx::fast::CustomKernelFunction &PartitionScatterKernel() {
	static auto kernel = []() {
		const std::string src = R"(
            uint gid = thread_position_in_grid.x;
            uint gsize = threads_per_grid.x;
            int nrows = int(keys_shape[0]);
            for (uint i = gid; i < uint(nrows); i += gsize) {
                const long k = keys[i];
                const long v = values[i];
                const ulong h = slotlock_hash(k);
                const uint p = slotlock_partition(h);
                const uint pos = atomic_fetch_add_explicit(&partition_write_pos[p], 1u, memory_order_relaxed);
                const uint out_idx = partition_offsets[p] + pos;
                scatter_keys[out_idx] = k;
                scatter_values[out_idx] = v;
            }
        )";
		return mx::fast::metal_kernel("mlx_gb_partition_scatter",
		                              {"keys", "values", "partition_offsets", "partition_write_pos"},
		                              {"scatter_keys", "scatter_values"}, src, SlotlockHashHeader(), true, false);
	}();
	return kernel;
}

const mx::fast::CustomKernelFunction &PartitionAggregateSlotlockKernel() {
	static auto kernel = []() {
		const std::string src = R"(
            const uint SLOT_EMPTY = 0u;
            const uint SLOT_LOCKED = 1u;
            const uint SLOT_COMMITTED = 2u;

            const uint tid = thread_position_in_threadgroup.x;
            const uint pid = threadgroup_position_in_grid.x;

            threadgroup atomic_uint slot_state[1024];
            threadgroup long slot_keys[1024];
            threadgroup long slot_sums[1024];

            for (uint s = tid; s < 1024u; s += 64u) {
                atomic_store_explicit(&slot_state[s], SLOT_EMPTY, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint slice_lo = partition_offsets[pid];
            const uint slice_hi = partition_offsets[pid + 1];

            for (uint i = slice_lo + tid; i < slice_hi; i += 64u) {
                const long k = scatter_keys[i];
                const long v = scatter_values[i];
                const ulong h = slotlock_hash(k);
                uint slot = slotlock_slot_start(h);
                for (uint attempt = 0; attempt < 4096u; ++attempt) {
                    const uint state = atomic_load_explicit(&slot_state[slot], memory_order_relaxed);
                    if (state == SLOT_EMPTY) {
                        uint expected = SLOT_EMPTY;
                        if (atomic_compare_exchange_weak_explicit(&slot_state[slot], &expected, SLOT_LOCKED,
                                                                  memory_order_relaxed, memory_order_relaxed)) {
                            slot_keys[slot] = k;
                            slot_sums[slot] = v;
                            atomic_store_explicit(&slot_state[slot], SLOT_COMMITTED, memory_order_relaxed);
                            break;
                        }
                        continue;
                    }
                    if (state == SLOT_COMMITTED) {
                        if (slot_keys[slot] == k) {
                            uint expected = SLOT_COMMITTED;
                            if (atomic_compare_exchange_weak_explicit(&slot_state[slot], &expected, SLOT_LOCKED,
                                                                      memory_order_relaxed, memory_order_relaxed)) {
                                slot_sums[slot] += v;
                                atomic_store_explicit(&slot_state[slot], SLOT_COMMITTED, memory_order_relaxed);
                                break;
                            }
                            continue;
                        }
                        slot = (slot + 1u) & 1023u;
                        continue;
                    }
                    continue;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint s = tid; s < 1024u; s += 64u) {
                const uint state = atomic_load_explicit(&slot_state[s], memory_order_relaxed);
                if (state == SLOT_COMMITTED) {
                    const uint out_idx = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
                    out_keys[out_idx] = slot_keys[s];
                    out_sums[out_idx] = slot_sums[s];
                }
            }
        )";
		return mx::fast::metal_kernel("mlx_gb_slotlock_agg", {"scatter_keys", "scatter_values", "partition_offsets"},
		                              {"out_count", "out_keys", "out_sums"}, src, SlotlockHashHeader(), true, false);
	}();
	return kernel;
}

void ExclusiveScanPartitions(const uint32_t *counts, uint32_t *offsets, int parts) {
	uint32_t running = 0;
	for (int p = 0; p < parts; p++) {
		offsets[p] = running;
		running += counts[p];
	}
	offsets[parts] = running;
}

int PickCountBlocks(int n) {
	return (n + kCountWork - 1) / kCountWork;
}

bool SlotlockChecksumOk(const mx::array &vals_i64, const std::vector<MlxGroupbyRow> &rows) {
	if (rows.empty()) {
		return false;
	}
	mx::eval(vals_i64);
	auto in_sum = mx::sum(vals_i64).item<int64_t>();
	int64_t out_sum = 0;
	for (auto &row : rows) {
		out_sum += static_cast<int64_t>(row.sum);
	}
	return in_sum == out_sum;
}

} // namespace

std::vector<MlxGroupbyRow> GroupbySumSlotlockGpu(mx::array keys, mx::array vals) {
	auto n = static_cast<int>(keys.shape(0));
	if (n == 0) {
		return {};
	}
	if (n > static_cast<int>(std::numeric_limits<uint32_t>::max())) {
		return {};
	}
	mx::eval({keys, vals});
	auto vals_i64 = mx::astype(mx::round(vals), mx::int64);

	const int parts = kSlotPartitions;
	auto partition_counts = mx::zeros({parts}, mx::uint32);
	auto partition_offsets = mx::zeros({parts + 1}, mx::uint32);
	auto partition_write_pos = mx::zeros({parts}, mx::uint32);
	auto scatter_keys = mx::zeros({n}, mx::int64);
	auto scatter_values = mx::zeros({n}, mx::int64);
	auto out_count = mx::zeros({1}, mx::uint32);
	auto out_keys = mx::zeros({n}, mx::int64);
	auto out_sums = mx::zeros({n}, mx::int64);

	const int count_blocks = PickCountBlocks(n);
	PartitionCountKernel()({keys}, {mx::Shape {parts}}, {mx::uint32}, std::make_tuple(count_blocks, 1, 1),
	                       std::make_tuple(kCountThreads, 1, 1), {}, 0.0f, false, mx::Device::gpu);
	mx::eval(partition_counts);

	auto counts_ptr = partition_counts.data<uint32_t>();
	std::vector<uint32_t> offsets_host(static_cast<size_t>(parts + 1));
	ExclusiveScanPartitions(counts_ptr, offsets_host.data(), parts);
	if (offsets_host[static_cast<size_t>(parts)] != static_cast<uint32_t>(n)) {
		return {};
	}
	partition_offsets = mx::array(offsets_host.data(), {parts + 1}, mx::uint32);
	partition_write_pos = mx::zeros({parts}, mx::uint32);

	PartitionScatterKernel()({keys, vals_i64, partition_offsets, partition_write_pos}, {mx::Shape {n}, mx::Shape {n}},
	                         {mx::int64, mx::int64}, std::make_tuple(count_blocks, 1, 1),
	                         std::make_tuple(kCountThreads, 1, 1), {}, 0.0f, false, mx::Device::gpu);

	PartitionAggregateSlotlockKernel()({scatter_keys, scatter_values, partition_offsets},
	                                   {mx::Shape {1}, mx::Shape {n}, mx::Shape {n}},
	                                   {mx::uint32, mx::int64, mx::int64}, std::make_tuple(kSlotPartitions, 1, 1),
	                                   std::make_tuple(kSlotTgThreads, 1, 1), {}, 0.0f, false, mx::Device::gpu);

	mx::eval({out_count, out_keys, out_sums});
	auto out_n = out_count.data<uint32_t>()[0];
	if (out_n == 0) {
		return {};
	}
	auto ok = out_keys.data<int64_t>();
	auto os = out_sums.data<int64_t>();
	std::vector<MlxGroupbyRow> rows(out_n);
	for (uint32_t i = 0; i < out_n; i++) {
		rows[i] = {ok[i], static_cast<double>(os[i]), 0};
	}
	std::sort(rows.begin(), rows.end(), [](const MlxGroupbyRow &a, const MlxGroupbyRow &b) { return a.key < b.key; });
	return rows;
}

bool GroupbySumSlotlockValid(mx::array keys, mx::array vals, const std::vector<MlxGroupbyRow> &rows) {
	auto vals_i64 = mx::astype(mx::round(vals), mx::int64);
	return SlotlockChecksumOk(vals_i64, rows);
}

} // namespace duckdb_mlx
