#include "mlx_bridge.hpp"

// This file must not include any DuckDB header (see mlx_bridge.hpp).
#include <mlx/mlx.h>

#include <numeric>
#include <vector>

namespace mx = mlx::core;

namespace duckdb_mlx {

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
