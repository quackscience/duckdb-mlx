#define DUCKDB_EXTENSION_MAIN

#include "duckdb_mlx_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/config.hpp"
#include "mlx_logger.hpp"

#ifdef DUCKDB_MLX_GPU_ENABLED
#include "mlx_bridge.hpp"
#endif

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
	auto info =
	    StringUtil::Format("duckdb_mlx gpu=%s mlx=%s spdlog=%s", MLX_GPU_AVAILABLE ? "available" : "unavailable",
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
	throw NotImplementedException("mlx_sum requires a GPU-enabled build of duckdb_mlx");
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
	config.AddExtensionOption("mlx_log_level", "Log verbosity of duckdb_mlx (trace|debug|info|warn|error|critical|off)",
	                          LogicalType::VARCHAR, Value("warn"), SetLogLevel);

	loader.RegisterFunction(ScalarFunction("mlx_info", {}, LogicalType::VARCHAR, MlxInfoFun));
	loader.RegisterFunction(ScalarFunction("mlx_selftest", {}, LogicalType::VARCHAR, MlxSelftestFun));
	loader.RegisterFunction(
	    ScalarFunction("mlx_sum", {LogicalType::LIST(LogicalType::BIGINT)}, LogicalType::BIGINT, MlxSumFun));

	duckdb_mlx::SetLogLevel("warn");
	duckdb_mlx::LogDebug(StringUtil::Format("duckdb_mlx loaded (gpu=%s)", MLX_GPU_AVAILABLE ? "true" : "false"));
}

void DuckdbMlxExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string DuckdbMlxExtension::Name() {
	return "duckdb_mlx";
}

std::string DuckdbMlxExtension::Version() const {
#ifdef EXT_VERSION_DUCKDB_MLX
	return EXT_VERSION_DUCKDB_MLX;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(duckdb_mlx, loader) {
	duckdb::LoadInternal(loader);
}
}
