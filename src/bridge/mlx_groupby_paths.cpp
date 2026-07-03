// GROUP BY path selection — SQL setting + env override (MLX_GROUPBY_PATH).

#include "mlx_groupby_detail.hpp"

#include <cstring>
#include <mutex>
#include <string>

namespace duckdb_mlx {

namespace {

std::mutex g_path_mutex;
std::string g_path_override;

} // namespace

void MlxGroupbySetPathOverride(const char *path) {
	std::lock_guard<std::mutex> guard(g_path_mutex);
	if (path == nullptr || path[0] == '\0') {
		g_path_override.clear();
		return;
	}
	g_path_override = path;
}

const char *MlxGroupbyPathOverride() {
	std::lock_guard<std::mutex> guard(g_path_mutex);
	return g_path_override.empty() ? nullptr : g_path_override.c_str();
}

const char *GroupbyPathFromEnv() {
	auto override_path = MlxGroupbyPathOverride();
	if (override_path != nullptr) {
		return override_path;
	}
	return std::getenv("MLX_GROUPBY_PATH");
}

bool GroupbyShouldTrySlotlock(int /*n*/, int64_t estimated_groups) {
	constexpr int64_t kMinGroups = 1024;
	constexpr int64_t kSafeCap = 16'000'000;

	const char *path = GroupbyPathFromEnv();
	if (path) {
		if (std::strcmp(path, "slotlock") == 0) {
			return true;
		}
		if (std::strcmp(path, "radix") == 0 || std::strcmp(path, "sort") == 0 || std::strcmp(path, "dense") == 0) {
			return false;
		}
	}
	if (estimated_groups > 0) {
		return estimated_groups >= kMinGroups && estimated_groups <= kSafeCap;
	}
	return false;
}

bool GroupbyShouldTryRadix(int n, int64_t estimated_groups) {
	constexpr int64_t kMinGroups = 1024;
	constexpr int64_t kSlotCap = 16'000'000;
	constexpr int kMinRows = 50'000;

	const char *path = GroupbyPathFromEnv();
	if (path) {
		if (std::strcmp(path, "radix") == 0) {
			return true;
		}
		if (std::strcmp(path, "dense") == 0 || std::strcmp(path, "sort") == 0 || std::strcmp(path, "slotlock") == 0) {
			return false;
		}
	}
	if (n < kMinRows) {
		return false;
	}
	if (estimated_groups > kSlotCap) {
		return true;
	}
	if (estimated_groups >= kMinGroups && estimated_groups <= kSlotCap) {
		return false;
	}
	return true;
}

} // namespace duckdb_mlx
