#define DUCKDB_EXTENSION_MAIN

#include "mlx_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/config.hpp"
#include "mlx_logger.hpp"
#include "mlx_vss.hpp"

#ifdef DUCKDB_MLX_GPU_ENABLED
#include "mlx_bridge.hpp"
#endif
#include "mlx_transparent.hpp"

namespace duckdb {

#ifdef DUCKDB_MLX_GPU_ENABLED
static constexpr bool MLX_GPU_AVAILABLE = true;
#else
static constexpr bool MLX_GPU_AVAILABLE = false;
#endif

static void MlxInfoFun(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DUCKDB_MLX_GPU_ENABLED
	auto mlx_version = duckdb_mlx::MlxVersion();
#else
	std::string mlx_version = "none";
#endif
	auto info = StringUtil::Format("mlx gpu=%s mlx=%s spdlog=%s", MLX_GPU_AVAILABLE ? "available" : "unavailable",
	                               mlx_version, duckdb_mlx::SpdlogVersion());
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto result_data = ConstantVector::GetData<string_t>(result);
	result_data[0] = StringVector::AddString(result, info);
}

static void MlxSelftestFun(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DUCKDB_MLX_GPU_ENABLED
	auto status = duckdb_mlx::MlxSelftest();
#else
	std::string status = "gpu-disabled";
#endif
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto result_data = ConstantVector::GetData<string_t>(result);
	result_data[0] = StringVector::AddString(result, status);
}

static void MlxStreamSumBenchFun(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DUCKDB_MLX_GPU_ENABLED
	auto count = args.size();
	UnifiedVectorFormat data;
	args.data[0].ToUnifiedFormat(count, data);
	auto strings = UnifiedVectorFormat::GetData<string_t>(data);
	std::string status;
	for (idx_t i = 0; i < count; i++) {
		auto row_idx = data.sel->get_index(i);
		if (!data.validity.RowIsValid(row_idx)) {
			status = "error=null key";
			break;
		}
		status = duckdb_mlx::MlxStreamSumBench(strings[row_idx].GetString());
	}
#else
	std::string status = "gpu-disabled";
#endif
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto result_data = ConstantVector::GetData<string_t>(result);
	result_data[0] = StringVector::AddString(result, status);
}

static void MlxMultiAggBenchFun(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DUCKDB_MLX_GPU_ENABLED
	auto count = args.size();
	UnifiedVectorFormat data;
	args.data[0].ToUnifiedFormat(count, data);
	auto strings = UnifiedVectorFormat::GetData<string_t>(data);
	std::string status;
	for (idx_t i = 0; i < count; i++) {
		auto row_idx = data.sel->get_index(i);
		if (!data.validity.RowIsValid(row_idx)) {
			status = "error=null key";
			break;
		}
		status = duckdb_mlx::MlxMultiAggBench(strings[row_idx].GetString());
	}
#else
	std::string status = "gpu-disabled";
#endif
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto result_data = ConstantVector::GetData<string_t>(result);
	result_data[0] = StringVector::AddString(result, status);
}

//! mlx_sum(BIGINT[]) — sums a list of BIGINTs on the GPU. Phase 0 spike
//! vehicle for moving DuckDB column data through the MLX bridge; the
//! transparent optimizer hook will replace it.
static void MlxSumFun(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DUCKDB_MLX_GPU_ENABLED
	auto count = args.size();
	auto &list_vec = args.data[0];

	UnifiedVectorFormat ldata;
	list_vec.ToUnifiedFormat(count, ldata);
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(ldata);

	auto &child = ListVector::GetEntry(list_vec);
	auto child_size = ListVector::GetListSize(list_vec);
	UnifiedVectorFormat cdata;
	child.ToUnifiedFormat(child_size, cdata);
	if (!cdata.validity.AllValid()) {
		throw NotImplementedException("mlx_sum does not support NULL list elements yet");
	}
	auto child_data = UnifiedVectorFormat::GetData<int64_t>(cdata);
	bool child_is_flat = cdata.sel == FlatVector::IncrementalSelectionVector();

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<int64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto row_idx = ldata.sel->get_index(i);
		if (!ldata.validity.RowIsValid(row_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}
		auto entry = list_entries[row_idx];
		if (child_is_flat) {
			// unified memory: hand the DuckDB buffer straight to the bridge
			result_data[i] = duckdb_mlx::MlxSumInt64(child_data + entry.offset, entry.length);
		} else {
			std::vector<int64_t> gathered(entry.length);
			for (idx_t j = 0; j < entry.length; j++) {
				gathered[j] = child_data[cdata.sel->get_index(entry.offset + j)];
			}
			result_data[i] = duckdb_mlx::MlxSumInt64(gathered.data(), gathered.size());
		}
	}
#else
	throw NotImplementedException("mlx_sum requires a GPU-enabled build of mlx");
#endif
}

//! mlx_expr_bench(BIGINT[]) — benchmark-only: ALU-dense fp32 expression on the
//! GPU (see MlxExprBenchInt64). Not bit-comparable with the CPU fp64 result.
static void MlxExprBenchFun(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DUCKDB_MLX_GPU_ENABLED
	auto count = args.size();
	auto &list_vec = args.data[0];

	UnifiedVectorFormat ldata;
	list_vec.ToUnifiedFormat(count, ldata);
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(ldata);

	auto &child = ListVector::GetEntry(list_vec);
	auto child_size = ListVector::GetListSize(list_vec);
	child.Flatten(child_size);
	auto child_data = FlatVector::GetData<int64_t>(child);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<double>(result);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto row_idx = ldata.sel->get_index(i);
		if (!ldata.validity.RowIsValid(row_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}
		auto entry = list_entries[row_idx];
		result_data[i] = duckdb_mlx::MlxExprBenchInt64(child_data + entry.offset, entry.length);
	}
#else
	throw NotImplementedException("mlx_expr_bench requires a GPU-enabled build of mlx");
#endif
}

//! mlx_cache_stats() — [segments_total, segments_pruned] from the last cached
//! GPU aggregate (partition-level zone-map pruning).
static void MlxCacheStatsFun(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DUCKDB_MLX_GPU_ENABLED
	auto stats = duckdb_mlx::MlxCacheLastStats();
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto list_data = ConstantVector::GetData<list_entry_t>(result);
	list_data[0].offset = 0;
	list_data[0].length = 2;
	auto &child = ListVector::GetEntry(result);
	child.SetVectorType(VectorType::FLAT_VECTOR);
	auto child_data = FlatVector::GetData<int64_t>(child);
	child_data[0] = stats.segments_total;
	child_data[1] = stats.segments_pruned;
	ListVector::SetListSize(result, 2);
#else
	throw NotImplementedException("mlx_cache_stats requires a GPU-enabled build of mlx");
#endif
}

//! mlx_cache_clear() — drops the entire GPU column cache (tests).
static void MlxCacheClearFun(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DUCKDB_MLX_GPU_ENABLED
	duckdb_mlx::MlxCacheClearAll();
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto result_data = ConstantVector::GetData<string_t>(result);
	result_data[0] = StringVector::AddString(result, "ok");
#else
	throw NotImplementedException("mlx_cache_clear requires a GPU-enabled build of mlx");
#endif
}

//! mlx_cache_pin(table) — GQE-style resident table: full scan into GPU column cache.
static void MlxCachePinFun(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DUCKDB_MLX_GPU_ENABLED
	auto table_name = args.data[0].GetValue(0).ToString();
	auto pin = MlxCachePinTable(state.GetContext(), table_name);
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto list_data = ConstantVector::GetData<list_entry_t>(result);
	list_data[0].offset = 0;
	list_data[0].length = 3;
	auto &child = ListVector::GetEntry(result);
	child.SetVectorType(VectorType::FLAT_VECTOR);
	auto child_data = FlatVector::GetData<int64_t>(child);
	child_data[0] = pin.rows;
	child_data[1] = pin.columns;
	child_data[2] = pin.already_resident ? 1 : 0;
	ListVector::SetListSize(result, 3);
#else
	throw NotImplementedException("mlx_cache_pin requires a GPU-enabled build of mlx");
#endif
}

//! mlx_cache_pin_tpch() — pin all eight TPC-H tables (GQE load_tpch.py equivalent).
static void MlxCachePinTpchFun(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DUCKDB_MLX_GPU_ENABLED
	MlxCachePinTpch(state.GetContext());
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto result_data = ConstantVector::GetData<string_t>(result);
	result_data[0] = StringVector::AddString(result, "ok");
#else
	throw NotImplementedException("mlx_cache_pin_tpch requires a GPU-enabled build of mlx");
#endif
}

//! mlx_groupby_bench(keys, values [, use_hash]) — GPU group-by sum spike; returns
//! checksum of per-group sums for timing comparisons vs CPU.
static void MlxGroupbyBenchFun(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DUCKDB_MLX_GPU_ENABLED
	auto count = args.size();
	auto &keys_vec = args.data[0];
	auto &vals_vec = args.data[1];
	bool use_hash = false;
	if (args.ColumnCount() >= 3) {
		use_hash = args.data[2].GetValue(0).GetValue<bool>();
	}

	UnifiedVectorFormat kdata;
	keys_vec.ToUnifiedFormat(count, kdata);
	auto key_entries = UnifiedVectorFormat::GetData<list_entry_t>(kdata);
	auto &key_child = ListVector::GetEntry(keys_vec);
	key_child.Flatten(ListVector::GetListSize(keys_vec));
	auto key_data = FlatVector::GetData<int64_t>(key_child);

	UnifiedVectorFormat vdata;
	vals_vec.ToUnifiedFormat(count, vdata);
	auto val_entries = UnifiedVectorFormat::GetData<list_entry_t>(vdata);
	auto &val_child = ListVector::GetEntry(vals_vec);
	val_child.Flatten(ListVector::GetListSize(vals_vec));
	auto val_data = FlatVector::GetData<int64_t>(val_child);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<double>(result);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto ki = kdata.sel->get_index(i);
		auto vi = vdata.sel->get_index(i);
		if (!kdata.validity.RowIsValid(ki) || !vdata.validity.RowIsValid(vi)) {
			result_validity.SetInvalid(i);
			continue;
		}
		auto ke = key_entries[ki];
		auto ve = val_entries[vi];
		vector<double> vals(ve.length);
		for (idx_t j = 0; j < ve.length; j++) {
			vals[j] = static_cast<double>(val_data[ve.offset + j]);
		}
		result_data[i] = duckdb_mlx::MlxGroupbyBenchSum(key_data + ke.offset, vals.data(), ke.length, use_hash);
	}
#else
	throw NotImplementedException("mlx_groupby_bench requires a GPU-enabled build of mlx");
#endif
}

static void SetLogLevel(ClientContext &context, SetScope scope, Value &parameter) {
	if (!duckdb_mlx::SetLogLevel(StringValue::Get(parameter))) {
		throw InvalidInputException("mlx_log_level must be one of trace|debug|info|warn|error|critical|off");
	}
}

static void LoadInternal(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());

	// GPU execution is opt-out where supported; on non-Apple-Silicon builds the
	// extension loads but leaves every plan on the CPU.
	config.AddExtensionOption("mlx_execution", "Execute supported query plans on the Apple GPU via Metal/MLX",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(MLX_GPU_AVAILABLE));
	config.AddExtensionOption("mlx_min_rows",
	                          "Minimum estimated scan cardinality before a plan is considered for GPU execution",
	                          LogicalType::UBIGINT, Value::UBIGINT(524288));
	config.AddExtensionOption("mlx_log_level", "Log verbosity of mlx (trace|debug|info|warn|error|critical|off)",
	                          LogicalType::VARCHAR, Value("warn"), SetLogLevel);

	loader.RegisterFunction(ScalarFunction("mlx_info", {}, LogicalType::VARCHAR, MlxInfoFun));
	loader.RegisterFunction(ScalarFunction("mlx_selftest", {}, LogicalType::VARCHAR, MlxSelftestFun));
	loader.RegisterFunction(
	    ScalarFunction("mlx_stream_sum_bench", {LogicalType::VARCHAR}, LogicalType::VARCHAR, MlxStreamSumBenchFun));
	loader.RegisterFunction(
	    ScalarFunction("mlx_multi_agg_bench", {LogicalType::VARCHAR}, LogicalType::VARCHAR, MlxMultiAggBenchFun));
	loader.RegisterFunction(
	    ScalarFunction("mlx_sum", {LogicalType::LIST(LogicalType::BIGINT)}, LogicalType::BIGINT, MlxSumFun));
	loader.RegisterFunction(ScalarFunction("mlx_expr_bench", {LogicalType::LIST(LogicalType::BIGINT)},
	                                       LogicalType::DOUBLE, MlxExprBenchFun));
	loader.RegisterFunction(
	    ScalarFunction("mlx_cache_stats", {}, LogicalType::LIST(LogicalType::BIGINT), MlxCacheStatsFun));
	loader.RegisterFunction(ScalarFunction("mlx_cache_clear", {}, LogicalType::VARCHAR, MlxCacheClearFun));
	loader.RegisterFunction(ScalarFunction("mlx_cache_pin", {LogicalType::VARCHAR},
	                                       LogicalType::LIST(LogicalType::BIGINT), MlxCachePinFun));
	loader.RegisterFunction(ScalarFunction("mlx_cache_pin_tpch", {}, LogicalType::VARCHAR, MlxCachePinTpchFun));
	loader.RegisterFunction(ScalarFunction(
	    "mlx_groupby_bench", {LogicalType::LIST(LogicalType::BIGINT), LogicalType::LIST(LogicalType::BIGINT)},
	    LogicalType::DOUBLE, MlxGroupbyBenchFun));
	loader.RegisterFunction(ScalarFunction(
	    "mlx_groupby_bench",
	    {LogicalType::LIST(LogicalType::BIGINT), LogicalType::LIST(LogicalType::BIGINT), LogicalType::BOOLEAN},
	    LogicalType::DOUBLE, MlxGroupbyBenchFun));
	RegisterMlxVss(loader);
#ifdef DUCKDB_MLX_GPU_ENABLED
	RegisterMlxOptimizer(loader.GetDatabaseInstance());
#endif

	duckdb_mlx::SetLogLevel("warn");
	duckdb_mlx::LogDebug(StringUtil::Format("mlx loaded (gpu=%s)", MLX_GPU_AVAILABLE ? "true" : "false"));
}

void MlxExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string MlxExtension::Name() {
	return "mlx";
}

std::string MlxExtension::Version() const {
#ifdef EXT_VERSION_MLX
	return EXT_VERSION_MLX;
#elif defined(EXT_VERSION_DUCKDB_MLX)
	return EXT_VERSION_DUCKDB_MLX;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(mlx, loader) {
	duckdb::LoadInternal(loader);
}
}
