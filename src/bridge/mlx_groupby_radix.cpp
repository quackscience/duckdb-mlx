// LSD radix sort GROUP BY — adapted from gpudb groupby.metal + metal_groupby.mm (v0.1.3).
// Uses GPU histogram + scan + atomic scatter; host segment-reduce.
// Apache-2.0 — https://github.com/singhpratech/duckdbgpumetaldbram

#include "mlx_groupby_detail.hpp"

#include <mlx/fast.h>
#include <mlx/mlx.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace mx = mlx::core;

namespace duckdb_mlx {

namespace {

constexpr int kRadixBuckets = 256;
constexpr int kRadixBlockSize = 256;
constexpr int kRadixWorkPerBlock = 1024;
constexpr int kSegThreads = 8;

const mx::fast::CustomKernelFunction &RadixFlipKernel() {
	static auto kernel = []() {
		const std::string src = R"(
            uint gid = thread_position_in_grid.x;
            int nrows = int(keys_shape[0]);
            if (gid >= uint(nrows)) {
                return;
            }
            keys[gid] = (long)((ulong)keys[gid] ^ 0x8000000000000000UL);
        )";
		return mx::fast::metal_kernel("mlx_gb_radix_flip", {"keys"}, {"keys"}, src, "", true, false);
	}();
	return kernel;
}

const mx::fast::CustomKernelFunction &RadixHistogramKernel() {
	static auto kernel = []() {
		const std::string src = R"(
            const uint WORK = 1024u;
            const uint BUCKETS = 256u;
            const uint PER_THREAD = 4u;
            uint tid = thread_position_in_threadgroup.x;
            uint bid = threadgroup_position_in_grid.x;
            int nrows = int(keys_shape[0]);
            uint n = uint(nrows);

            threadgroup atomic_uint local_hist[256];
            if (tid < BUCKETS) {
                atomic_store_explicit(&local_hist[tid], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint base = bid * WORK;
            const uint shift = shift_buf[0];
            const uint dst = tid * PER_THREAD;
            for (uint j = 0; j < PER_THREAD; ++j) {
                const uint idx = base + dst + j;
                if (idx >= n) {
                    break;
                }
                const ulong k = (ulong)keys[idx];
                const uint bucket = (uint)((k >> shift) & 0xFFu);
                atomic_fetch_add_explicit(&local_hist[bucket], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid < BUCKETS) {
                const uint count = atomic_load_explicit(&local_hist[tid], memory_order_relaxed);
                atomic_store_explicit(&hist[bid * BUCKETS + tid], count, memory_order_relaxed);
            }
        )";
		return mx::fast::metal_kernel("mlx_gb_radix_histogram", {"keys", "shift_buf"}, {"hist"}, src, "", true, false);
	}();
	return kernel;
}

const mx::fast::CustomKernelFunction &RadixScatterKernel() {
	static auto kernel = []() {
		const std::string src = R"(
            const uint WORK = 1024u;
            const uint BUCKETS = 256u;
            uint tid = thread_position_in_threadgroup.x;
            uint bid = threadgroup_position_in_grid.x;
            int nrows = int(in_keys_shape[0]);
            uint n = uint(nrows);

            threadgroup atomic_uint pos[256];
            if (tid < BUCKETS) {
                atomic_store_explicit(&pos[tid], scan[bid * BUCKETS + tid], memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint base = bid * WORK;
            const uint shift = shift_buf[0];
            const uint gsize = 256u;
            for (uint i = base + tid; i < min(base + WORK, n); i += gsize) {
                const ulong k = (ulong)in_keys[i];
                const uint b = (uint)((k >> shift) & 0xFFu);
                const uint dst = atomic_fetch_add_explicit(&pos[b], 1u, memory_order_relaxed);
                out_keys[dst] = in_keys[i];
                out_values[dst] = in_values[i];
            }
        )";
		return mx::fast::metal_kernel("mlx_gb_radix_scatter", {"in_keys", "in_values", "shift_buf", "scan"},
		                              {"out_keys", "out_values"}, src, "", true, false);
	}();
	return kernel;
}

const mx::fast::CustomKernelFunction &RadixBucketTotalsKernel() {
	static auto kernel = []() {
		const std::string src = R"(
            const uint BUCKETS = 256u;
            uint tid = thread_position_in_threadgroup.x;
            uint b = threadgroup_position_in_grid.x;
            int num_blocks = int(num_blocks_buf[0]);
            threadgroup uint shm[256];
            uint running = 0;
            for (int chunk = 0; chunk < num_blocks; chunk += 256) {
                int bid = chunk + int(tid);
                running += (bid < num_blocks) ? hist[bid * BUCKETS + b] : 0u;
            }
            shm[tid] = running;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) {
                    shm[tid] += shm[tid + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                bucket_total[b] = shm[0];
            }
        )";
		return mx::fast::metal_kernel("mlx_gb_radix_bucket_totals", {"hist", "num_blocks_buf"}, {"bucket_total"}, src,
		                              "", true, false);
	}();
	return kernel;
}

const mx::fast::CustomKernelFunction &RadixBucketOffsetsKernel() {
	static auto kernel = []() {
		const std::string src = R"(
            uint tid = thread_position_in_threadgroup.x;
            threadgroup uint shm[256];
            shm[tid] = bucket_total[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint offset = 1; offset < 256; offset <<= 1) {
                uint t = (tid >= offset) ? shm[tid - offset] : 0u;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                shm[tid] += t;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            bucket_offset[tid] = shm[tid] - bucket_total[tid];
        )";
		return mx::fast::metal_kernel("mlx_gb_radix_bucket_offsets", {"bucket_total"}, {"bucket_offset"}, src, "", true,
		                              false);
	}();
	return kernel;
}

const mx::fast::CustomKernelFunction &RadixPerBucketScanKernel() {
	static auto kernel = []() {
		const std::string src = R"(
            const uint BUCKETS = 256u;
            uint tid = thread_position_in_threadgroup.x;
            uint b = threadgroup_position_in_grid.x;
            int num_blocks = int(num_blocks_buf[0]);
            threadgroup uint shm[256];
            uint running = bucket_offset[b];
            for (int chunk = 0; chunk < num_blocks; chunk += 256) {
                int bid = chunk + int(tid);
                uint val = (bid < num_blocks) ? hist[bid * BUCKETS + b] : 0u;
                shm[tid] = val;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint offset = 1; offset < 256; offset <<= 1) {
                    uint t = (tid >= offset) ? shm[tid - offset] : 0u;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    shm[tid] += t;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (bid < num_blocks) {
                    scan[bid * BUCKETS + b] = running + (shm[tid] - val);
                }
                running += shm[255];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        )";
		return mx::fast::metal_kernel("mlx_gb_radix_per_bucket_scan",
		                              {"hist", "num_blocks_buf", "bucket_offset"}, {"scan"}, src, "", true, false);
	}();
	return kernel;
}

mx::array GpuBuildRadixScan(mx::array hist, uint32_t num_blocks) {
	std::vector<uint32_t> nb = {num_blocks};
	auto num_blocks_buf = mx::array(nb.data(), {1}, mx::uint32);

	auto bucket_total = RadixBucketTotalsKernel()({hist, num_blocks_buf}, {mx::Shape {kRadixBuckets}}, {mx::uint32},
	                                              std::make_tuple(kRadixBuckets, 1, 1),
	                                              std::make_tuple(kRadixBlockSize, 1, 1), {}, 0.0f, false,
	                                              mx::Device::gpu)[0];
	auto bucket_offset = RadixBucketOffsetsKernel()({bucket_total}, {mx::Shape {kRadixBuckets}}, {mx::uint32},
	                                                std::make_tuple(1, 1, 1), std::make_tuple(kRadixBlockSize, 1, 1),
	                                                {}, 0.0f, false, mx::Device::gpu)[0];
	auto scan = RadixPerBucketScanKernel()({hist, num_blocks_buf, bucket_offset}, {hist.shape()}, {mx::uint32},
	                                       std::make_tuple(kRadixBuckets, 1, 1), std::make_tuple(kRadixBlockSize, 1, 1),
	                                       {}, 0.0f, false, mx::Device::gpu)[0];
	mx::eval(scan);
	return scan;
}

void BuildRadixScanHost(const uint32_t *hist, uint32_t num_blocks, uint32_t *scan) {
	std::vector<uint32_t> bucket_total(kRadixBuckets, 0);
	for (uint32_t bid = 0; bid < num_blocks; bid++) {
		for (int b = 0; b < kRadixBuckets; b++) {
			bucket_total[static_cast<size_t>(b)] += hist[static_cast<size_t>(bid) * kRadixBuckets + b];
		}
	}
	std::vector<uint32_t> bucket_offset(kRadixBuckets);
	uint32_t run = 0;
	for (int b = 0; b < kRadixBuckets; b++) {
		bucket_offset[static_cast<size_t>(b)] = run;
		run += bucket_total[static_cast<size_t>(b)];
	}
	for (int b = 0; b < kRadixBuckets; b++) {
		uint32_t running = bucket_offset[static_cast<size_t>(b)];
		for (uint32_t bid = 0; bid < num_blocks; bid++) {
			scan[static_cast<size_t>(bid) * kRadixBuckets + b] = running;
			running += hist[static_cast<size_t>(bid) * kRadixBuckets + b];
		}
	}
}

uint8_t ActiveRadixBytes(int64_t mn, int64_t mx) {
	const uint64_t mn_u = static_cast<uint64_t>(mn) ^ 0x8000000000000000ULL;
	const uint64_t mx_u = static_cast<uint64_t>(mx) ^ 0x8000000000000000ULL;
	uint8_t active = 0;
	for (int p = 0; p < 8; p++) {
		const uint64_t shift = static_cast<uint64_t>(p) * 8;
		const uint8_t mn_b = static_cast<uint8_t>((mn_u >> shift) & 0xFFu);
		const uint8_t mx_b = static_cast<uint8_t>((mx_u >> shift) & 0xFFu);
		if (mn_b != mx_b) {
			active |= static_cast<uint8_t>(1u << p);
		}
	}
	return active == 0 ? 1 : active;
}

std::vector<MlxGroupbyRow> HostSegmentReduce(const int64_t *sk, const int64_t *sv, size_t n) {
	if (n == 0) {
		return {};
	}
	if (n < 16384) {
		std::vector<MlxGroupbyRow> out;
		out.reserve(64);
		size_t i = 0;
		while (i < n) {
			const int64_t k = sk[i];
			int64_t sum = 0;
			size_t j = i;
			while (j < n && sk[j] == k) {
				sum += sv[j];
				++j;
			}
			out.push_back({k, static_cast<double>(sum), static_cast<int64_t>(j - i)});
			i = j;
		}
		return out;
	}

	std::vector<std::vector<int64_t>> tk(kSegThreads), ts(kSegThreads);
	std::array<size_t, kSegThreads + 1> starts;
	starts[0] = 0;
	for (int t = 1; t < kSegThreads; t++) {
		size_t s = (n / kSegThreads) * static_cast<size_t>(t);
		while (s < n && s > 0 && sk[s] == sk[s - 1]) {
			++s;
		}
		starts[static_cast<size_t>(t)] = s;
	}
	starts[static_cast<size_t>(kSegThreads)] = n;

	std::vector<std::thread> workers;
	workers.reserve(kSegThreads);
	for (int t = 0; t < kSegThreads; t++) {
		workers.emplace_back([&, t] {
			const size_t lo = starts[static_cast<size_t>(t)];
			const size_t hi = starts[static_cast<size_t>(t + 1)];
			size_t i = lo;
			while (i < hi) {
				const int64_t k = sk[i];
				int64_t sum = 0;
				size_t j = i;
				while (j < hi && sk[j] == k) {
					sum += sv[j];
					++j;
				}
				tk[static_cast<size_t>(t)].push_back(k);
				ts[static_cast<size_t>(t)].push_back(sum);
				i = j;
			}
		});
	}
	for (auto &w : workers) {
		w.join();
	}

	std::vector<MlxGroupbyRow> out;
	for (int t = 0; t < kSegThreads; t++) {
		for (size_t i = 0; i < tk[static_cast<size_t>(t)].size(); i++) {
			out.push_back({tk[static_cast<size_t>(t)][i], static_cast<double>(ts[static_cast<size_t>(t)][i]), 0});
		}
	}
	std::sort(out.begin(), out.end(), [](const MlxGroupbyRow &a, const MlxGroupbyRow &b) { return a.key < b.key; });
	return out;
}

bool RadixChecksumOk(const mx::array &vals_i64, const std::vector<MlxGroupbyRow> &rows) {
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

std::vector<MlxGroupbyRow> GroupbySumRadixGpu(mx::array keys, mx::array vals) {
	auto n = static_cast<int>(keys.shape(0));
	if (n == 0) {
		return {};
	}
	if (n > static_cast<int>(std::numeric_limits<uint32_t>::max())) {
		return {};
	}
	mx::eval({keys, vals});
	auto vals_i64 = mx::astype(mx::round(vals), mx::int64);

	const uint32_t n32 = static_cast<uint32_t>(n);
	const uint32_t num_blocks = (n32 + kRadixWorkPerBlock - 1) / kRadixWorkPerBlock;
	const size_t hist_elems = static_cast<size_t>(num_blocks) * kRadixBuckets;

	auto keys_a = keys;
	auto keys_b = mx::zeros({n}, mx::int64);
	auto vals_a = vals_i64;
	auto vals_b = mx::zeros({n}, mx::int64);

	// Sign-flip for signed radix passes.
	RadixFlipKernel()({keys_a}, {mx::Shape {n}}, {mx::int64},
	                  std::make_tuple((n32 + kRadixBlockSize - 1) / kRadixBlockSize, 1, 1),
	                  std::make_tuple(kRadixBlockSize, 1, 1), {}, 0.0f, false, mx::Device::gpu);
	mx::eval(keys_a);

	mx::eval(keys_a);
	auto kmin = mx::min(keys_a).item<int64_t>();
	auto kmax = mx::max(keys_a).item<int64_t>();
	const uint8_t active_bytes = ActiveRadixBytes(kmin, kmax);

	mx::array *in_keys = &keys_a;
	mx::array *in_vals = &vals_a;
	mx::array *out_keys = &keys_b;
	mx::array *out_vals = &vals_b;

	for (int pass = 0; pass < 8; pass++) {
		if (!(active_bytes & (1u << pass))) {
			continue;
		}
		const uint32_t shift = static_cast<uint32_t>(pass) * 8u;
		auto shift_buf = mx::array({static_cast<int>(shift)}, mx::uint32);
		auto hist = mx::zeros({static_cast<int>(hist_elems)}, mx::uint32);

		RadixHistogramKernel()({*in_keys, shift_buf}, {mx::Shape {static_cast<int>(hist_elems)}}, {mx::uint32},
		                       std::make_tuple(num_blocks, 1, 1), std::make_tuple(kRadixBlockSize, 1, 1), {}, 0.0f,
		                       false, mx::Device::gpu);
		mx::eval(hist);

		auto scan = GpuBuildRadixScan(hist, num_blocks);

		RadixScatterKernel()({*in_keys, *in_vals, shift_buf, scan}, {mx::Shape {n}, mx::Shape {n}},
		                     {mx::int64, mx::int64}, std::make_tuple(num_blocks, 1, 1),
		                     std::make_tuple(kRadixBlockSize, 1, 1), {}, 0.0f, false, mx::Device::gpu);
		mx::eval({*out_keys, *out_vals});
		std::swap(in_keys, out_keys);
		std::swap(in_vals, out_vals);
	}

	RadixFlipKernel()({*in_keys}, {mx::Shape {n}}, {mx::int64},
	                  std::make_tuple((n32 + kRadixBlockSize - 1) / kRadixBlockSize, 1, 1),
	                  std::make_tuple(kRadixBlockSize, 1, 1), {}, 0.0f, false, mx::Device::gpu);
	mx::eval(*in_keys);

	auto sk = in_keys->data<int64_t>();
	auto sv = in_vals->data<int64_t>();
	return HostSegmentReduce(sk, sv, static_cast<size_t>(n));
}

bool GroupbySumRadixValid(mx::array keys, mx::array vals, const std::vector<MlxGroupbyRow> &rows) {
	auto vals_i64 = mx::astype(mx::round(vals), mx::int64);
	return RadixChecksumOk(vals_i64, rows);
}

} // namespace duckdb_mlx
