#include "mlx_logger.hpp"

// spdlog linked through vcpkg. This file must not include any DuckDB header
// (see mlx_logger.hpp).
#include <spdlog/spdlog.h>

namespace duckdb_mlx {

bool SetLogLevel(const std::string &level) {
	auto parsed = spdlog::level::from_str(level);
	if (parsed == spdlog::level::off && level != "off") {
		return false;
	}
	spdlog::set_level(parsed);
	return true;
}

std::string SpdlogVersion() {
	return std::to_string(SPDLOG_VER_MAJOR) + "." + std::to_string(SPDLOG_VER_MINOR) + "." +
	       std::to_string(SPDLOG_VER_PATCH);
}

void LogTrace(const std::string &msg) {
	spdlog::trace(msg);
}
void LogDebug(const std::string &msg) {
	spdlog::debug(msg);
}
void LogInfo(const std::string &msg) {
	spdlog::info(msg);
}
void LogWarn(const std::string &msg) {
	spdlog::warn(msg);
}
void LogError(const std::string &msg) {
	spdlog::error(msg);
}

} // namespace duckdb_mlx
