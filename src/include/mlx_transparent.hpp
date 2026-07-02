#pragma once

#include "duckdb/common/common.hpp"

namespace duckdb {

class ClientContext;
class DatabaseInstance;

struct MlxCachePinResult {
	int64_t rows = 0;
	int64_t columns = 0;
	bool already_resident = false;
};

//! GQE-style resident table: full unfiltered scan of every pin-able column into
//! the GPU column cache (analogous to GQE load_tpch.py COPY into GPU memory).
MlxCachePinResult MlxCachePinTable(ClientContext &context, const string &table_name);

//! Pin all eight standard TPC-H tables (customer, lineitem, …).
void MlxCachePinTpch(ClientContext &context);

//! Registers the optimizer extension that transparently routes supported
//! ungrouped SUM aggregations to the GPU (PLAN Phase 1). Only called on
//! GPU-enabled builds.
void RegisterMlxOptimizer(DatabaseInstance &db);

} // namespace duckdb
