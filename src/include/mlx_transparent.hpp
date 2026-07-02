#pragma once

namespace duckdb {

class DatabaseInstance;

//! Registers the optimizer extension that transparently routes supported
//! ungrouped SUM aggregations to the GPU (PLAN Phase 1). Only called on
//! GPU-enabled builds.
void RegisterMlxOptimizer(DatabaseInstance &db);

} // namespace duckdb
