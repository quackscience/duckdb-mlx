#include "mlx_bridge.hpp"
#include "mlx_groupby_detail.hpp"

#include <mlx/compile.h>
#include <mlx/fast.h>
#include <mlx/mlx.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mx = mlx::core;

namespace duckdb_mlx {

namespace {

constexpr int64_t kDenseMaxSpan = 1'048'576; // perfect-hash window (matches DuckDB low-cardinality path)

struct GroupbyHashKernels {
	mx::fast::CustomKernelFunction build;
	mx::fast::CustomKernelFunction collect;
};

const GroupbyHashKernels &HashKernels() {
	static GroupbyHashKernels kernels = []() {
		const std::string build_src = R"(
            uint idx = thread_position_in_grid.x;
            if (idx >= keys_shape[0]) {
                return;
            }
            int64_t key = keys[idx];
            float val = values[idx];
            uint cap = static_cast<uint>(capacity[0]);
            uint h = static_cast<uint>(key ^ (key >> 33));
            h = (h * 2654435761u) % cap;
            for (uint probe = 0; probe < cap; ++probe) {
                uint slot = (h + probe) % cap;
                int expected = 0;
                if (atomic_compare_exchange_weak_explicit(
                        &slot_used[slot], &expected, 1, memory_order_relaxed, memory_order_relaxed)) {
                    hash_keys[slot] = key;
                    hash_sums[slot] = val;
                    hash_counts[slot] = 1;
                    return;
                }
                if (hash_keys[slot] == key) {
                    atomic_fetch_add_explicit(&hash_sums[slot], val, memory_order_relaxed);
                    atomic_fetch_add_explicit(&hash_counts[slot], 1, memory_order_relaxed);
                    return;
                }
            }
        )";

		const std::string collect_src = R"(
            uint slot = thread_position_in_grid.x;
            if (slot >= hash_keys_shape[0]) {
                return;
            }
            if (slot_used[slot] == 0) {
                out_keys[slot] = 0;
                out_sums[slot] = 0.0f;
                out_counts[slot] = 0;
                return;
            }
            out_keys[slot] = hash_keys[slot];
            out_sums[slot] = hash_sums[slot];
            out_counts[slot] = hash_counts[slot];
        )";

		return GroupbyHashKernels {
		    mx::fast::metal_kernel("mlx_groupby_build", {"keys", "values", "capacity"},
		                           {"hash_keys", "hash_sums", "hash_counts", "slot_used"}, build_src, "", true, true),
		    mx::fast::metal_kernel("mlx_groupby_collect",
		                           {"hash_keys", "hash_sums", "hash_counts", "slot_used"},
		                           {"out_keys", "out_sums", "out_counts"}, collect_src, "", true, false),
		};
	}();
	return kernels;
}

uint32_t NextPow2(uint32_t x) {
	if (x <= 1) {
		return 1;
	}
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x + 1;
}

mx::array KeysFromCacheColumn(const mx::array &col) {
	return mx::astype(mx::round(col), mx::int64);
}

bool DenseSpanOk(int64_t kmin, int64_t kmax) {
	if (kmin == INT64_MIN || kmax == INT64_MIN) {
		return false;
	}
	if (kmax < kmin) {
		return false;
	}
	auto span = static_cast<uint64_t>(kmax - kmin + 1);
	return span > 0 && span <= static_cast<uint64_t>(kDenseMaxSpan);
}

std::vector<MlxGroupbyRow> DenseRowsFromTables(int64_t kmin, mx::array sums, mx::array counts) {
	auto span = static_cast<int>(sums.shape(0));
	mx::eval({sums, counts});
	auto sums_ptr = sums.data<float>();
	auto counts_ptr = counts.data<int32_t>();
	std::vector<MlxGroupbyRow> out;
	out.reserve(span);
	for (int i = 0; i < span; i++) {
		if (counts_ptr[i] > 0) {
			out.push_back({kmin + i, static_cast<double>(sums_ptr[i]), counts_ptr[i]});
		}
	}
	return out;
}

//! PERFECT_HASH-style: O(n) scatter into dense[key - min] table. No sort.
std::vector<MlxGroupbyRow> GroupbySumDenseGpu(mx::array key_arr, mx::array val_arr) {
	auto n = key_arr.shape(0);
	if (n == 0) {
		return {};
	}
	mx::eval({key_arr, val_arr});
	auto kmin = mx::min(key_arr).item<int64_t>();
	auto kmax = mx::max(key_arr).item<int64_t>();
	if (!DenseSpanOk(kmin, kmax)) {
		return {};
	}
	auto span = static_cast<int>(kmax - kmin + 1);
	auto idx = mx::astype(mx::subtract(key_arr, mx::array(kmin, mx::int64)), mx::int32);
	auto sums = mx::zeros({span}, mx::float32);
	auto counts = mx::zeros({span}, mx::int32);
	auto val_up = mx::expand_dims(val_arr, 1);
	auto one_up = mx::expand_dims(mx::astype(mx::ones_like(val_arr), mx::int32), 1);
	sums = mx::scatter_add(sums, idx, val_up, 0);
	counts = mx::scatter_add(counts, idx, one_up, 0);
	return DenseRowsFromTables(kmin, sums, counts);
}

//! CPU dense accumulate in double — accurate cold path, still O(n) not O(n log n).
std::vector<MlxGroupbyRow> GroupbySumDenseHost(const int64_t *keys, const double *values, size_t n) {
	if (n == 0) {
		return {};
	}
	int64_t kmin = INT64_MAX;
	int64_t kmax = INT64_MIN;
	for (size_t i = 0; i < n; i++) {
		auto key = keys[i];
		if (key == INT64_MIN) {
			continue;
		}
		kmin = std::min(kmin, key);
		kmax = std::max(kmax, key);
	}
	if (!DenseSpanOk(kmin, kmax)) {
		return {};
	}
	auto span = static_cast<size_t>(kmax - kmin + 1);
	std::vector<double> sums(span, 0.0);
	std::vector<int64_t> counts(span, 0);
	for (size_t i = 0; i < n; i++) {
		auto key = keys[i];
		if (key == INT64_MIN) {
			continue;
		}
		auto slot = static_cast<size_t>(key - kmin);
		sums[slot] += values[i];
		counts[slot]++;
	}
	std::vector<MlxGroupbyRow> out;
	out.reserve(span);
	for (size_t i = 0; i < span; i++) {
		if (counts[i] > 0) {
			out.push_back({kmin + static_cast<int64_t>(i), sums[i], counts[i]});
		}
	}
	return out;
}

std::vector<MlxGroupbyRow> GroupbySumSortAccurate(mx::array key_arr, const double *host_vals, size_t n) {
	if (n == 0) {
		return {};
	}
	auto perm = mx::argsort(key_arr);
	mx::eval(perm);
	auto perm_ptr = perm.data<int32_t>();
	auto key_ptr = key_arr.data<int64_t>();

	std::vector<MlxGroupbyRow> out;
	int64_t cur_key = key_ptr[perm_ptr[0]];
	double acc = host_vals[perm_ptr[0]];
	int cnt = 1;
	for (size_t i = 1; i < n; i++) {
		auto src = perm_ptr[i];
		auto key = key_ptr[src];
		if (key != cur_key) {
			out.push_back({cur_key, acc, cnt});
			cur_key = key;
			acc = host_vals[src];
			cnt = 1;
		} else {
			acc += host_vals[src];
			cnt++;
		}
	}
	out.push_back({cur_key, acc, cnt});
	return out;
}

std::vector<MlxGroupbyRow> GroupbySumHash(mx::array keys, mx::array vals) {
	auto n = static_cast<int>(keys.shape(0));
	if (n == 0) {
		return {};
	}
	uint32_t cap = NextPow2(static_cast<uint32_t>(std::max<size_t>(n * 2, 1024)));
	std::vector<int32_t> cap_host = {static_cast<int32_t>(cap)};
	mx::array capacity(cap_host.begin(), mx::Shape {1}, mx::int32);

	auto &kernels = HashKernels();
	auto built = kernels.build({keys, vals, capacity},
	                           {mx::Shape {static_cast<int>(cap)}, mx::Shape {static_cast<int>(cap)},
	                            mx::Shape {static_cast<int>(cap)}, mx::Shape {static_cast<int>(cap)}},
	                           {mx::int64, mx::float32, mx::int32, mx::int32}, std::make_tuple(n, 1, 1),
	                           std::make_tuple(256, 1, 1), {}, 0.0f, false, mx::Device::gpu);

	auto hash_keys = built[0];
	auto hash_sums = built[1];
	auto hash_counts = built[2];
	auto slot_used = built[3];

	auto collected = kernels.collect({hash_keys, hash_sums, hash_counts, slot_used},
	                                 {mx::Shape {static_cast<int>(cap)}, mx::Shape {static_cast<int>(cap)},
	                                  mx::Shape {static_cast<int>(cap)}},
	                                 {mx::int64, mx::float32, mx::int32}, std::make_tuple(static_cast<int>(cap), 1, 1),
	                                 std::make_tuple(256, 1, 1), {}, std::nullopt, false, mx::Device::gpu);

	mx::eval(collected);
	auto out_keys = collected[0].data<int64_t>();
	auto out_sums = collected[1].data<float>();
	auto out_counts = collected[2].data<int32_t>();
	auto used = slot_used.data<int32_t>();

	std::vector<MlxGroupbyRow> rows;
	for (uint32_t i = 0; i < cap; i++) {
		if (used[i]) {
			rows.push_back({out_keys[i], static_cast<double>(out_sums[i]), out_counts[i]});
		}
	}
	std::sort(rows.begin(), rows.end(), [](const MlxGroupbyRow &a, const MlxGroupbyRow &b) { return a.key < b.key; });
	return rows;
}

std::vector<MlxGroupbyRow> PickGpuPath(mx::array key_arr, mx::array val_arr, bool use_hash) {
	if (use_hash) {
		return GroupbySumHash(key_arr, val_arr);
	}
	auto dense = GroupbySumDenseGpu(key_arr, val_arr);
	if (!dense.empty() || key_arr.shape(0) == 0) {
		return dense;
	}
	// high-cardinality fallback: sort + scatter (no host perm walk)
	auto n = static_cast<int>(key_arr.shape(0));
	if (n <= 1) {
		mx::eval({key_arr, val_arr});
		return {{key_arr.item<int64_t>(), static_cast<double>(val_arr.item<float>()), 1}};
	}
	auto perm = mx::argsort(key_arr);
	auto sk = mx::take(key_arr, perm, 0);
	auto sv = mx::take(val_arr, perm, 0);
	auto prev = mx::slice(sk, {0}, {n - 1});
	auto curr = mx::slice(sk, {1}, {n});
	auto diff = mx::astype(mx::not_equal(curr, prev), mx::int32);
	auto boundary = mx::concatenate({mx::array({1}, mx::int32), diff}, 0);
	auto group_ids = mx::cumsum(boundary, 0);
	int g = mx::add(mx::max(group_ids), mx::array({1}, mx::int32)).item<int32_t>();
	if (g <= 0) {
		return {};
	}
	auto sums = mx::zeros({g}, mx::float32);
	sums = mx::scatter_add(sums, group_ids, mx::expand_dims(sv, 1), 0);
	auto counts = mx::zeros({g}, mx::int32);
	counts = mx::scatter_add(counts, group_ids, mx::expand_dims(mx::astype(mx::ones_like(sv), mx::int32), 1), 0);
	mx::eval({boundary, sums, counts, sk});
	auto boundary_ptr = boundary.data<int32_t>();
	std::vector<int32_t> boundary_idx;
	boundary_idx.reserve(g);
	for (int i = 0; i < n; i++) {
		if (boundary_ptr[i]) {
			boundary_idx.push_back(i);
		}
	}
	mx::array idx_arr(boundary_idx.data(), {static_cast<int>(boundary_idx.size())}, mx::int32);
	auto group_keys = mx::take(sk, idx_arr, 0);
	mx::eval(group_keys);
	auto gk = group_keys.data<int64_t>();
	auto sums_ptr = sums.data<float>();
	auto counts_ptr = counts.data<int32_t>();
	std::vector<MlxGroupbyRow> out(boundary_idx.size());
	for (size_t i = 0; i < boundary_idx.size(); i++) {
		out[i] = {gk[i], static_cast<double>(sums_ptr[i]), counts_ptr[i]};
	}
	return out;
}

} // namespace

std::vector<MlxGroupbyRow> MlxGroupbySumSortAccurate(mx::array key_arr, const double *host_vals, size_t n) {
	return GroupbySumSortAccurate(key_arr, host_vals, n);
}

std::vector<MlxGroupbyRow> MlxGroupbySumArrays(mx::array key_arr, mx::array val_arr, bool use_hash) {
	return PickGpuPath(key_arr, val_arr, use_hash);
}

std::vector<MlxGroupbyRow> MlxGroupbySumDenseGpuArrays(mx::array key_arr, mx::array val_arr) {
	return GroupbySumDenseGpu(key_arr, val_arr);
}

std::vector<MlxGroupbyRow> MlxGroupbyDenseTable(int64_t kmin, mx::array sums, mx::array counts) {
	return DenseRowsFromTables(kmin, sums, counts);
}

namespace {

struct DenseGroupbyAcc {
	int64_t population = -1;
	int64_t kmin = 0;
	mx::array sums = mx::zeros({0}, mx::float32);
	mx::array counts = mx::zeros({0}, mx::int32);
	bool ready = false;
};

std::unordered_map<std::string, std::unique_ptr<DenseGroupbyAcc>> &DenseGroupbyStore() {
	static std::unordered_map<std::string, std::unique_ptr<DenseGroupbyAcc>> store;
	return store;
}

DenseGroupbyAcc &DenseGroupbyEntry(const std::string &dk) {
	auto &slot = DenseGroupbyStore()[dk];
	if (!slot) {
		slot = std::make_unique<DenseGroupbyAcc>();
	}
	return *slot;
}

std::string DenseGroupbyKey(const std::string &group_key, const std::string &value_key) {
	return group_key + '\0' + value_key;
}

void DenseGroupbyResize(DenseGroupbyAcc &acc, int64_t kmin, int64_t kmax) {
	auto new_span = static_cast<int>(kmax - kmin + 1);
	if (!acc.ready) {
		acc.kmin = kmin;
		acc.sums = mx::zeros({new_span}, mx::float32);
		acc.counts = mx::zeros({new_span}, mx::int32);
		acc.ready = true;
		return;
	}
	auto old_span = static_cast<int>(acc.sums.shape(0));
	auto old_kmin = acc.kmin;
	auto old_kmax = old_kmin + old_span - 1;
	auto merged_kmin = std::min(kmin, old_kmin);
	auto merged_kmax = std::max(kmax, old_kmax);
	auto merged_span = static_cast<int>(merged_kmax - merged_kmin + 1);
	if (merged_kmin == old_kmin && merged_span == old_span) {
		return;
	}
	auto new_sums = mx::zeros({merged_span}, mx::float32);
	auto new_counts = mx::zeros({merged_span}, mx::int32);
	auto off = static_cast<int>(old_kmin - merged_kmin);
	new_sums = mx::scatter_add(new_sums, mx::array(off, mx::int32), mx::expand_dims(acc.sums, 1), 0);
	new_counts = mx::scatter_add(new_counts, mx::array(off, mx::int32), mx::expand_dims(acc.counts, 1), 0);
	acc.kmin = merged_kmin;
	acc.sums = new_sums;
	acc.counts = new_counts;
}

} // namespace

void MlxGroupbyDenseAccumulate(const std::string &group_col_key, const std::string &value_col_key, int64_t population,
                               mx::array key_arr, mx::array val_arr) {
	if (key_arr.shape(0) == 0) {
		return;
	}
	auto dk = DenseGroupbyKey(group_col_key, value_col_key);
	auto &acc = DenseGroupbyEntry(dk);
	if (acc.population != population) {
		acc.population = population;
		acc.kmin = 0;
		acc.ready = false;
	}
	mx::eval({key_arr, val_arr});
	auto kmin = mx::min(key_arr).item<int64_t>();
	auto kmax = mx::max(key_arr).item<int64_t>();
	if (!DenseSpanOk(kmin, kmax)) {
		return;
	}
	DenseGroupbyResize(acc, kmin, kmax);
	auto idx = mx::astype(mx::subtract(key_arr, mx::array(acc.kmin, mx::int64)), mx::int32);
	acc.sums = mx::scatter_add(acc.sums, idx, mx::expand_dims(val_arr, 1), 0);
	acc.counts = mx::scatter_add(acc.counts, idx, mx::expand_dims(mx::astype(mx::ones_like(val_arr), mx::int32), 1), 0);
	mx::eval({acc.sums, acc.counts});
}

bool MlxGroupbyDenseTryRead(const std::string &group_col_key, const std::string &value_col_key, int64_t population,
                            std::vector<MlxGroupbyRow> &out) {
	auto it = DenseGroupbyStore().find(DenseGroupbyKey(group_col_key, value_col_key));
	if (it == DenseGroupbyStore().end() || !it->second || !it->second->ready ||
	    it->second->population != population) {
		return false;
	}
	out = DenseRowsFromTables(it->second->kmin, it->second->sums, it->second->counts);
	return true;
}

void MlxGroupbyDenseClearTable(const std::string &table_prefix) {
	for (auto it = DenseGroupbyStore().begin(); it != DenseGroupbyStore().end();) {
		if (table_prefix.empty()) {
			it = DenseGroupbyStore().erase(it);
			continue;
		}
		auto pos = it->first.find('\0');
		auto group_key = pos == std::string::npos ? it->first : it->first.substr(0, pos);
		if (group_key.compare(0, table_prefix.size(), table_prefix) == 0) {
			it = DenseGroupbyStore().erase(it);
		} else {
			++it;
		}
	}
}

std::vector<MlxGroupbyRow> MlxGroupbySum(const int64_t *keys, const double *values, const uint8_t *valid, size_t count,
                                        bool use_hash) {
	if (count == 0) {
		return {};
	}
	if (!use_hash) {
		auto dense = GroupbySumDenseHost(keys, values, count);
		if (!dense.empty()) {
			return dense;
		}
	}
	mx::array key_arr(keys, mx::Shape {static_cast<int>(count)}, mx::int64);
	std::vector<float> vals_f32(count);
	for (size_t i = 0; i < count; i++) {
		vals_f32[i] = static_cast<float>(values[i]);
	}
	mx::array val_arr(vals_f32.data(), mx::Shape {static_cast<int>(count)}, mx::float32);
	if (valid) {
		mx::array mask(valid, mx::Shape {static_cast<int>(count)}, mx::uint8);
		auto m = mx::astype(mask, mx::bool_);
		key_arr = mx::where(m, key_arr, mx::array(INT64_MIN));
		val_arr = mx::where(m, val_arr, mx::array(0.0f));
	}
	return PickGpuPath(key_arr, val_arr, use_hash);
}

double MlxGroupbyBenchSum(const int64_t *keys, const double *values, size_t count, bool use_hash) {
	auto rows = MlxGroupbySum(keys, values, nullptr, count, use_hash);
	double total = 0;
	for (auto &row : rows) {
		total += row.sum;
	}
	return total;
}

} // namespace duckdb_mlx
