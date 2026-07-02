#include "mlx_vss.hpp"

#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "mlx_bridge.hpp"

namespace duckdb {

//! mlx_vss_pin(name, list(<FLOAT[N] column>) [, precision]) — copies the
//! embedding column into a named GPU-resident, L2-normalized matrix stored as
//! 'float32' (default) or 'float16'. Returns the row count. Row i of the
//! pinned matrix is list position i, so pin with a stable order
//! (e.g. list(emb ORDER BY id)).
static void MlxVssPinFun(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DUCKDB_MLX_GPU_ENABLED
	auto count = args.size();
	auto &name_vec = args.data[0];
	auto &list_vec = args.data[1];

	bool half = false;
	if (args.ColumnCount() == 3) {
		auto precision = args.data[2].GetValue(0).ToString();
		if (precision == "float16") {
			half = true;
		} else if (precision != "float32") {
			throw InvalidInputException("mlx_vss_pin precision must be 'float32' or 'float16', got '%s'", precision);
		}
	}

	auto &type = list_vec.GetType();
	if (type.id() != LogicalTypeId::LIST || ListType::GetChildType(type).id() != LogicalTypeId::ARRAY ||
	    ArrayType::GetChildType(ListType::GetChildType(type)).id() != LogicalTypeId::FLOAT) {
		throw InvalidInputException("mlx_vss_pin expects list(<FLOAT[N] column>), got %s", type.ToString());
	}
	auto dim = ArrayType::GetSize(ListType::GetChildType(type));

	UnifiedVectorFormat ndata;
	name_vec.ToUnifiedFormat(count, ndata);
	auto names = UnifiedVectorFormat::GetData<string_t>(ndata);

	UnifiedVectorFormat ldata;
	list_vec.ToUnifiedFormat(count, ldata);
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(ldata);

	auto &array_vec = ListVector::GetEntry(list_vec);
	auto array_count = ListVector::GetListSize(list_vec);
	array_vec.Flatten(array_count);
	if (!FlatVector::Validity(array_vec).AllValid()) {
		throw InvalidInputException("mlx_vss_pin does not support NULL embeddings");
	}
	auto &float_vec = ArrayVector::GetEntry(array_vec);
	float_vec.Flatten(array_count * dim);
	if (!FlatVector::Validity(float_vec).AllValid()) {
		throw InvalidInputException("mlx_vss_pin does not support NULL embedding elements");
	}
	auto float_data = FlatVector::GetData<float>(float_vec);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<int64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto name_idx = ndata.sel->get_index(i);
		auto list_idx = ldata.sel->get_index(i);
		if (!ndata.validity.RowIsValid(name_idx) || !ldata.validity.RowIsValid(list_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}
		auto entry = list_entries[list_idx];
		result_data[i] = duckdb_mlx::MlxVssPin(names[name_idx].GetString(), float_data + entry.offset * dim,
		                                       entry.length, dim, half);
	}
#else
	throw NotImplementedException("mlx_vss_pin requires a GPU-enabled build of duckdb_mlx");
#endif
}

struct MlxVssSearchBindData : public TableFunctionData {
	string name;
	vector<float> query;
	int64_t k = 0;
};

struct MlxVssSearchState : public GlobalTableFunctionState {
	vector<duckdb_mlx::MlxVssMatch> matches;
	idx_t offset = 0;
};

static unique_ptr<FunctionData> MlxVssSearchBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<MlxVssSearchBindData>();
	bind_data->name = StringValue::Get(input.inputs[0]);
	for (auto &v : ListValue::GetChildren(input.inputs[1])) {
		bind_data->query.push_back(v.GetValue<float>());
	}
	bind_data->k = input.inputs[2].GetValue<int64_t>();
	return_types = {LogicalType::BIGINT, LogicalType::FLOAT};
	names = {"idx", "score"};
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> MlxVssSearchInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<MlxVssSearchState>();
#ifdef DUCKDB_MLX_GPU_ENABLED
	auto &bind_data = input.bind_data->Cast<MlxVssSearchBindData>();
	try {
		state->matches = duckdb_mlx::MlxVssSearch(bind_data.name, bind_data.query.data(),
		                                          NumericCast<int64_t>(bind_data.query.size()), bind_data.k);
	} catch (const std::runtime_error &ex) {
		throw InvalidInputException("mlx_vss_search: %s", ex.what());
	}
#else
	throw NotImplementedException("mlx_vss_search requires a GPU-enabled build of duckdb_mlx");
#endif
	return std::move(state);
}

static void MlxVssSearchFun(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<MlxVssSearchState>();
	auto idx_data = FlatVector::GetData<int64_t>(output.data[0]);
	auto score_data = FlatVector::GetData<float>(output.data[1]);
	idx_t out = 0;
	while (state.offset < state.matches.size() && out < STANDARD_VECTOR_SIZE) {
		idx_data[out] = state.matches[state.offset].index;
		score_data[out] = state.matches[state.offset].score;
		out++;
		state.offset++;
	}
	output.SetCardinality(out);
}

struct MlxVssBatchBindData : public TableFunctionData {
	string name;
	vector<float> queries; // row-major Q x dim
	int64_t q = 0;
	int64_t dim = 0;
	int64_t k = 0;
};

struct MlxVssBatchState : public GlobalTableFunctionState {
	vector<duckdb_mlx::MlxVssBatchMatch> matches;
	idx_t offset = 0;
};

static unique_ptr<FunctionData> MlxVssBatchBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<MlxVssBatchBindData>();
	bind_data->name = StringValue::Get(input.inputs[0]);
	for (auto &query : ListValue::GetChildren(input.inputs[1])) {
		auto &values = ListValue::GetChildren(query);
		if (bind_data->dim == 0) {
			bind_data->dim = NumericCast<int64_t>(values.size());
		} else if (bind_data->dim != NumericCast<int64_t>(values.size())) {
			throw InvalidInputException("mlx_vss_search_batch: all query vectors must have the same length");
		}
		for (auto &v : values) {
			bind_data->queries.push_back(v.GetValue<float>());
		}
		bind_data->q++;
	}
	bind_data->k = input.inputs[2].GetValue<int64_t>();
	return_types = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::FLOAT};
	names = {"query_no", "idx", "score"};
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> MlxVssBatchInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<MlxVssBatchState>();
#ifdef DUCKDB_MLX_GPU_ENABLED
	auto &bind_data = input.bind_data->Cast<MlxVssBatchBindData>();
	try {
		state->matches = duckdb_mlx::MlxVssSearchBatch(bind_data.name, bind_data.queries.data(), bind_data.q,
		                                               bind_data.dim, bind_data.k);
	} catch (const std::runtime_error &ex) {
		throw InvalidInputException("mlx_vss_search_batch: %s", ex.what());
	}
#else
	throw NotImplementedException("mlx_vss_search_batch requires a GPU-enabled build of duckdb_mlx");
#endif
	return std::move(state);
}

static void MlxVssBatchFun(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<MlxVssBatchState>();
	auto query_data = FlatVector::GetData<int64_t>(output.data[0]);
	auto idx_data = FlatVector::GetData<int64_t>(output.data[1]);
	auto score_data = FlatVector::GetData<float>(output.data[2]);
	idx_t out = 0;
	while (state.offset < state.matches.size() && out < STANDARD_VECTOR_SIZE) {
		query_data[out] = state.matches[state.offset].query;
		idx_data[out] = state.matches[state.offset].index;
		score_data[out] = state.matches[state.offset].score;
		out++;
		state.offset++;
	}
	output.SetCardinality(out);
}

void RegisterMlxVss(ExtensionLoader &loader) {
	ScalarFunctionSet pin_set("mlx_vss_pin");
	pin_set.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::ANY}, LogicalType::BIGINT, MlxVssPinFun));
	pin_set.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::ANY, LogicalType::VARCHAR},
	                                   LogicalType::BIGINT, MlxVssPinFun));
	loader.RegisterFunction(pin_set);

	TableFunction search("mlx_vss_search",
	                     {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::BIGINT},
	                     MlxVssSearchFun, MlxVssSearchBind, MlxVssSearchInit);
	loader.RegisterFunction(search);

	TableFunction batch(
	    "mlx_vss_search_batch",
	    {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::LIST(LogicalType::FLOAT)), LogicalType::BIGINT},
	    MlxVssBatchFun, MlxVssBatchBind, MlxVssBatchInit);
	loader.RegisterFunction(batch);
}

} // namespace duckdb
