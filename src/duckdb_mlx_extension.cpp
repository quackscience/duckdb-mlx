#define DUCKDB_EXTENSION_MAIN

#include "duckdb_mlx_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/config.hpp"
#include "mlx_logger.hpp"

namespace duckdb {

#ifdef DUCKDB_MLX_GPU_ENABLED
static constexpr bool MLX_GPU_AVAILABLE = true;
#else
static constexpr bool MLX_GPU_AVAILABLE = false;
#endif

static void MlxInfoFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto info = StringUtil::Format("duckdb_mlx gpu=%s spdlog=%s", MLX_GPU_AVAILABLE ? "available" : "unavailable",
	                               duckdb_mlx::SpdlogVersion());
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	auto result_data = ConstantVector::GetData<string_t>(result);
	result_data[0] = StringVector::AddString(result, info);
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
