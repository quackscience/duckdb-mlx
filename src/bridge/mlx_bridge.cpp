#include "mlx_bridge.hpp"

// This file must not include any DuckDB header (see mlx_bridge.hpp).
#include <mlx/mlx.h>

#include <algorithm>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mx = mlx::core;

namespace duckdb_mlx {

namespace {
std::mutex vss_mutex;
// Process-lifetime GPU-resident cache of L2-normalized embedding matrices.
std::unordered_map<std::string, mx::array> &VssStore() {
	static std::unordered_map<std::string, mx::array> store;
	return store;
}
} // namespace

int64_t MlxVssPin(const std::string &name, const float *data, int64_t n, int64_t dim, bool half) {
	mx::array m(data, mx::Shape {static_cast<int>(n), static_cast<int>(dim)}, mx::float32);
	auto norms = mx::sqrt(mx::sum(mx::square(m), {1}, true));
	auto normalized = mx::divide(m, mx::maximum(norms, mx::array(1e-12f)));
	if (half) {
		normalized = mx::astype(normalized, mx::float16);
	}
	normalized.eval();
	std::lock_guard<std::mutex> guard(vss_mutex);
	VssStore().erase(name);
	VssStore().emplace(name, std::move(normalized));
	return n;
}

std::vector<MlxVssMatch> MlxVssSearch(const std::string &name, const float *query, int64_t dim, int64_t k) {
	mx::array m = [&] {
		std::lock_guard<std::mutex> guard(vss_mutex);
		auto it = VssStore().find(name);
		if (it == VssStore().end()) {
			throw std::runtime_error("no pinned matrix named '" + name + "' — call mlx_vss_pin first");
		}
		return it->second;
	}();
	if (m.shape(1) != dim) {
		throw std::runtime_error("query dimension " + std::to_string(dim) + " does not match pinned dimension " +
		                         std::to_string(m.shape(1)));
	}
	auto n = m.shape(0);
	k = std::min<int64_t>(k, n);
	if (k <= 0) {
		return {};
	}

	mx::array q(query, mx::Shape {static_cast<int>(dim)}, mx::float32);
	auto qn = mx::divide(q, mx::maximum(mx::sqrt(mx::sum(mx::square(q))), mx::array(1e-12f)));
	auto scores = mx::astype(mx::matmul(m, mx::astype(qn, m.dtype())), mx::float32);
	// argpartition is O(N); the k selected rows are ordered on the host
	auto part = mx::argpartition(mx::negative(scores), static_cast<int>(k - 1), 0);
	mx::eval({part, scores});

	auto part_ptr = part.data<uint32_t>();
	auto scores_ptr = scores.data<float>();
	std::vector<MlxVssMatch> matches;
	matches.reserve(k);
	for (int64_t i = 0; i < k; i++) {
		matches.push_back({part_ptr[i], scores_ptr[part_ptr[i]]});
	}
	std::sort(matches.begin(), matches.end(), [](const MlxVssMatch &a, const MlxVssMatch &b) {
		return a.score != b.score ? a.score > b.score : a.index < b.index;
	});
	return matches;
}

std::string MlxVersion() {
	return mx::version();
}

std::string MlxSelftest() {
	try {
		if (!mx::metal::is_available()) {
			return "metal unavailable";
		}
		if (mx::default_device().type != mx::Device::gpu) {
			return "default device is not gpu";
		}
		// int64 reduction: sum(0..99999) == 4999950000
		std::vector<int64_t> seq(100000);
		std::iota(seq.begin(), seq.end(), 0);
		mx::array a(seq.begin(), mx::Shape {static_cast<int>(seq.size())}, mx::int64);
		auto sum = mx::sum(a).item<int64_t>();
		if (sum != 4999950000LL) {
			return "int64 sum mismatch: got " + std::to_string(sum);
		}
		// float32 elementwise + reduction: sum(2*2 x 1024) == 4096
		auto b = mx::full({1024}, 2.0f);
		auto sq = mx::sum(mx::multiply(b, b)).item<float>();
		if (sq != 4096.0f) {
			return "float32 multiply/sum mismatch: got " + std::to_string(sq);
		}
		return "ok";
	} catch (const std::exception &ex) {
		return std::string("exception: ") + ex.what();
	}
}

std::vector<MlxVssBatchMatch> MlxVssSearchBatch(const std::string &name, const float *queries, int64_t q, int64_t dim,
                                                int64_t k) {
	mx::array m = [&] {
		std::lock_guard<std::mutex> guard(vss_mutex);
		auto it = VssStore().find(name);
		if (it == VssStore().end()) {
			throw std::runtime_error("no pinned matrix named '" + name + "' — call mlx_vss_pin first");
		}
		return it->second;
	}();
	if (m.shape(1) != dim) {
		throw std::runtime_error("query dimension " + std::to_string(dim) + " does not match pinned dimension " +
		                         std::to_string(m.shape(1)));
	}
	auto n = m.shape(0);
	k = std::min<int64_t>(k, n);
	if (k <= 0 || q <= 0) {
		return {};
	}

	mx::array qm(queries, mx::Shape {static_cast<int>(q), static_cast<int>(dim)}, mx::float32);
	auto qnorms = mx::sqrt(mx::sum(mx::square(qm), {1}, true));
	auto qn = mx::divide(qm, mx::maximum(qnorms, mx::array(1e-12f)));
	auto scores = mx::astype(mx::matmul(mx::astype(qn, m.dtype()), mx::transpose(m)), mx::float32); // (Q, N)
	// argpartition is O(N) per row; each row's k selected entries are ordered
	// on the host
	auto part = mx::argpartition(mx::negative(scores), static_cast<int>(k - 1), 1);
	mx::eval({part, scores});

	auto part_ptr = part.data<uint32_t>();
	auto scores_ptr = scores.data<float>();
	std::vector<MlxVssBatchMatch> matches;
	matches.reserve(q * k);
	std::vector<MlxVssMatch> row;
	for (int64_t qi = 0; qi < q; qi++) {
		row.clear();
		for (int64_t i = 0; i < k; i++) {
			auto idx = part_ptr[qi * n + i];
			row.push_back({idx, scores_ptr[qi * n + idx]});
		}
		std::sort(row.begin(), row.end(), [](const MlxVssMatch &a, const MlxVssMatch &b) {
			return a.score != b.score ? a.score > b.score : a.index < b.index;
		});
		for (auto &match : row) {
			matches.push_back({qi, match.index, match.score});
		}
	}
	return matches;
}

double MlxExprBenchInt64(const int64_t *data, size_t count) {
	if (count == 0) {
		return 0;
	}
	mx::array a(data, mx::Shape {static_cast<int>(count)}, mx::int64);
	auto f = mx::astype(a, mx::float32);
	auto expr = mx::add(mx::multiply(mx::sin(f), mx::cos(f)), mx::sqrt(mx::add(mx::abs(f), mx::array(1.0f))));
	return static_cast<double>(mx::sum(expr).item<float>());
}

int64_t MlxSumInt64(const int64_t *data, size_t count) {
	if (count == 0) {
		return 0;
	}
	mx::array a(data, mx::Shape {static_cast<int>(count)}, mx::int64);
	return mx::sum(a).item<int64_t>();
}

} // namespace duckdb_mlx
