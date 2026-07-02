#pragma once

#include <string>

namespace duckdb_mlx {

// Thin wrapper around spdlog. spdlog pulls in libfmt, whose headers share
// include guards and include paths with DuckDB's vendored fmt (namespace
// duckdb_fmt), so no translation unit may include both duckdb.hpp and
// spdlog.h. All logging goes through this interface instead.

//! Set the global log level; returns false if the level name is invalid.
bool SetLogLevel(const std::string &level);
std::string SpdlogVersion();

void LogTrace(const std::string &msg);
void LogDebug(const std::string &msg);
void LogInfo(const std::string &msg);
void LogWarn(const std::string &msg);
void LogError(const std::string &msg);

} // namespace duckdb_mlx
