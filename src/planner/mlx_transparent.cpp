#include "mlx_transparent.hpp"

#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_transaction.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/storage_index.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/null_filter.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "mlx_bridge.hpp"
#include "mlx_logger.hpp"

#include <set>

namespace duckdb {

using duckdb_mlx::MlxAggKind;
using duckdb_mlx::MlxColumnData;
using duckdb_mlx::MlxExprOp;
using duckdb_mlx::MlxExprOpCode;
using duckdb_mlx::MlxFilter;
using duckdb_mlx::MlxSumProgram;

//===--------------------------------------------------------------------===//
// Expression translation: DuckDB expression tree -> postfix IR over the
// output columns of the child LogicalGet. Any unsupported node declines the
// whole plan (fallback is sacred).
//===--------------------------------------------------------------------===//
struct MlxExprTranslator {
	explicit MlxExprTranslator(LogicalGet &get, optional_ptr<LogicalProjection> proj) : get(get), proj(proj) {
	}

	LogicalGet &get;
	optional_ptr<LogicalProjection> proj;
	vector<MlxExprOp> ops;
	std::set<int32_t> referenced_cols;

	enum class Lane : uint8_t { FLOAT_LANE, INT_LANE, BOOL_LANE };

	//! DECIMAL(<=18) evaluates exactly in the int64 lane as raw scaled integers
	static bool IsIntLaneType(const LogicalType &type) {
		return type.id() == LogicalTypeId::DECIMAL && type.InternalType() != PhysicalType::INT128;
	}

	static bool IsSupportedColumnType(const LogicalType &type) {
		switch (type.id()) {
		case LogicalTypeId::DOUBLE:
		case LogicalTypeId::FLOAT:
		case LogicalTypeId::BIGINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::DATE: // epoch days ride the fp32 lane exactly (< 2^24)
			return true;
		case LogicalTypeId::DECIMAL:
			return IsIntLaneType(type);
		default:
			return false;
		}
	}

	static int64_t DecimalRaw(const Value &value) {
		switch (value.type().InternalType()) {
		case PhysicalType::INT16:
			return value.GetValueUnsafe<int16_t>();
		case PhysicalType::INT32:
			return value.GetValueUnsafe<int32_t>();
		default:
			return value.GetValueUnsafe<int64_t>();
		}
	}

	static int64_t Pow10(uint8_t exponent) {
		int64_t result = 1;
		for (uint8_t i = 0; i < exponent; i++) {
			result *= 10;
		}
		return result;
	}

	bool Translate(const Expression &expr, Lane &lane) {
		switch (expr.GetExpressionClass()) {
		case ExpressionClass::BOUND_COLUMN_REF: {
			auto &colref = expr.Cast<BoundColumnRefExpression>();
			if (proj && colref.binding.table_index == proj->table_index) {
				return Translate(*proj->expressions[colref.binding.column_index], lane);
			}
			if (colref.binding.table_index != get.table_index) {
				return false;
			}
			if (!IsSupportedColumnType(colref.return_type)) {
				return false;
			}
			auto &column_ids = get.GetColumnIds();
			auto col_idx = colref.binding.column_index;
			if (col_idx >= column_ids.size()) {
				return false;
			}
			auto storage_col = NumericCast<int32_t>(column_ids[col_idx].GetPrimaryIndex());
			ops.push_back({MlxExprOpCode::LOAD_COL, storage_col, 0});
			referenced_cols.insert(storage_col);
			lane = IsIntLaneType(colref.return_type) ? Lane::INT_LANE : Lane::FLOAT_LANE;
			return true;
		}
		case ExpressionClass::BOUND_CONSTANT: {
			auto &constant = expr.Cast<BoundConstantExpression>();
			if (constant.value.IsNull()) {
				return false;
			}
			auto &type = constant.value.type();
			if (type.id() == LogicalTypeId::DECIMAL) {
				if (!IsIntLaneType(type)) {
					return false;
				}
				MlxExprOp op {MlxExprOpCode::CONST_VAL, 0, 0};
				op.ivalue = DecimalRaw(constant.value);
				op.int_lane = true;
				ops.push_back(op);
				lane = Lane::INT_LANE;
				return true;
			}
			if (type.id() == LogicalTypeId::DATE) {
				ops.push_back(
				    {MlxExprOpCode::CONST_VAL, 0, static_cast<double>(constant.value.GetValue<date_t>().days)});
				lane = Lane::FLOAT_LANE;
				return true;
			}
			if (!type.IsNumeric()) {
				return false;
			}
			ops.push_back({MlxExprOpCode::CONST_VAL, 0, constant.value.GetValue<double>()});
			lane = Lane::FLOAT_LANE;
			return true;
		}
		case ExpressionClass::BOUND_CAST: {
			auto &cast = expr.Cast<BoundCastExpression>();
			if (cast.try_cast) {
				return false;
			}
			auto &to = cast.return_type;
			auto &from = cast.child->return_type;
			if (to.id() == LogicalTypeId::DECIMAL && from.id() == LogicalTypeId::DECIMAL) {
				// widening rescale = exact integer multiply by 10^k
				if (!IsIntLaneType(to) || !IsIntLaneType(from)) {
					return false;
				}
				auto k = static_cast<int>(DecimalType::GetScale(to)) - static_cast<int>(DecimalType::GetScale(from));
				if (k < 0) {
					return false;
				}
				Lane child_lane;
				if (!Translate(*cast.child, child_lane) || child_lane != Lane::INT_LANE) {
					return false;
				}
				if (k > 0) {
					MlxExprOp op {MlxExprOpCode::CONST_VAL, 0, 0};
					op.ivalue = Pow10(NumericCast<uint8_t>(k));
					op.int_lane = true;
					ops.push_back(op);
					ops.push_back({MlxExprOpCode::MUL, 0, 0});
				}
				lane = Lane::INT_LANE;
				return true;
			}
			if (to.id() != LogicalTypeId::DOUBLE && to.id() != LogicalTypeId::FLOAT) {
				return false;
			}
			if (from.id() == LogicalTypeId::DECIMAL) {
				if (!IsIntLaneType(from)) {
					return false;
				}
				Lane child_lane;
				if (!Translate(*cast.child, child_lane) || child_lane != Lane::INT_LANE) {
					return false;
				}
				auto scale = DecimalType::GetScale(from);
				ops.push_back({MlxExprOpCode::TO_FLOAT, 0, 1.0 / static_cast<double>(Pow10(scale))});
				lane = Lane::FLOAT_LANE;
				return true;
			}
			if (!from.IsNumeric()) {
				return false;
			}
			Lane child_lane;
			if (!Translate(*cast.child, child_lane) || child_lane != Lane::FLOAT_LANE) {
				return false;
			}
			lane = Lane::FLOAT_LANE;
			return true;
		}
		case ExpressionClass::BOUND_COMPARISON: {
			auto &comparison = expr.Cast<BoundComparisonExpression>();
			MlxExprOpCode code;
			switch (expr.GetExpressionType()) {
			case ExpressionType::COMPARE_LESSTHAN:
				code = MlxExprOpCode::CMP_LT;
				break;
			case ExpressionType::COMPARE_LESSTHANOREQUALTO:
				code = MlxExprOpCode::CMP_LE;
				break;
			case ExpressionType::COMPARE_GREATERTHAN:
				code = MlxExprOpCode::CMP_GT;
				break;
			case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
				code = MlxExprOpCode::CMP_GE;
				break;
			case ExpressionType::COMPARE_EQUAL:
				code = MlxExprOpCode::CMP_EQ;
				break;
			case ExpressionType::COMPARE_NOTEQUAL:
				code = MlxExprOpCode::CMP_NE;
				break;
			default:
				return false;
			}
			Lane left_lane;
			Lane right_lane;
			if (!Translate(*comparison.left, left_lane) || !Translate(*comparison.right, right_lane)) {
				return false;
			}
			if (left_lane != right_lane || left_lane == Lane::BOOL_LANE) {
				return false;
			}
			ops.push_back({code, 0, 0});
			lane = Lane::BOOL_LANE;
			return true;
		}
		case ExpressionClass::BOUND_CONJUNCTION: {
			auto &conjunction = expr.Cast<BoundConjunctionExpression>();
			auto code =
			    expr.GetExpressionType() == ExpressionType::CONJUNCTION_AND ? MlxExprOpCode::AND : MlxExprOpCode::OR;
			Lane child_lane;
			if (!Translate(*conjunction.children[0], child_lane) || child_lane != Lane::BOOL_LANE) {
				return false;
			}
			for (idx_t i = 1; i < conjunction.children.size(); i++) {
				if (!Translate(*conjunction.children[i], child_lane) || child_lane != Lane::BOOL_LANE) {
					return false;
				}
				ops.push_back({code, 0, 0});
			}
			lane = Lane::BOOL_LANE;
			return true;
		}
		case ExpressionClass::BOUND_OPERATOR: {
			auto &op = expr.Cast<BoundOperatorExpression>();
			if (expr.GetExpressionType() != ExpressionType::OPERATOR_NOT || op.children.size() != 1) {
				return false;
			}
			Lane child_lane;
			if (!Translate(*op.children[0], child_lane) || child_lane != Lane::BOOL_LANE) {
				return false;
			}
			ops.push_back({MlxExprOpCode::NOT, 0, 0});
			lane = Lane::BOOL_LANE;
			return true;
		}
		case ExpressionClass::BOUND_FUNCTION: {
			auto &function = expr.Cast<BoundFunctionExpression>();
			auto &name = function.function.name;
			if (function.children.size() == 2) {
				MlxExprOpCode code;
				if (name == "+") {
					code = MlxExprOpCode::ADD;
				} else if (name == "-") {
					code = MlxExprOpCode::SUB;
				} else if (name == "*") {
					code = MlxExprOpCode::MUL;
				} else if (name == "/") {
					code = MlxExprOpCode::DIV;
				} else {
					return false;
				}
				Lane left_lane;
				Lane right_lane;
				if (!Translate(*function.children[0], left_lane) || !Translate(*function.children[1], right_lane)) {
					return false;
				}
				if (left_lane != right_lane || left_lane == Lane::BOOL_LANE) {
					return false;
				}
				if (left_lane == Lane::INT_LANE && code == MlxExprOpCode::DIV) {
					return false; // integer division truncates; the CPU handles it
				}
				ops.push_back({code, 0, 0});
				lane = left_lane;
				return true;
			}
			if (function.children.size() == 1) {
				MlxExprOpCode code;
				bool float_only = false;
				if (name == "-") {
					code = MlxExprOpCode::NEGATE;
				} else if (name == "sin") {
					code = MlxExprOpCode::SIN;
					float_only = true;
				} else if (name == "cos") {
					code = MlxExprOpCode::COS;
					float_only = true;
				} else if (name == "sqrt") {
					code = MlxExprOpCode::SQRT;
					float_only = true;
				} else if (name == "abs") {
					code = MlxExprOpCode::ABS;
				} else {
					return false;
				}
				Lane child_lane;
				if (!Translate(*function.children[0], child_lane) || child_lane == Lane::BOOL_LANE) {
					return false;
				}
				if (float_only && child_lane != Lane::FLOAT_LANE) {
					return false;
				}
				ops.push_back({code, 0, 0});
				lane = child_lane;
				return true;
			}
			return false;
		}
		default:
			return false;
		}
	}
};

template <class T, class OUT>
static void AppendMlxColumnT(UnifiedVectorFormat &fmt, idx_t count, vector<OUT> &values, vector<uint8_t> &valid) {
	auto data = UnifiedVectorFormat::GetData<T>(fmt);
	auto base = values.size();
	values.resize(base + count);
	if (!fmt.sel->IsSet() && fmt.validity.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			values[base + i] = static_cast<OUT>(data[i]);
		}
		if (!valid.empty()) {
			valid.resize(base + count, 1);
		}
		return;
	}
	bool chunk_has_nulls = !fmt.validity.AllValid();
	if (chunk_has_nulls || !valid.empty()) {
		valid.resize(base + count, 1);
	}
	for (idx_t i = 0; i < count; i++) {
		auto idx = fmt.sel->get_index(i);
		bool row_valid = fmt.validity.RowIsValid(idx);
		values[base + i] = row_valid ? static_cast<OUT>(data[idx]) : static_cast<OUT>(0);
		if (!row_valid) {
			valid[base + i] = 0;
		}
	}
}

template <class T>
static void AppendMlxColumn(UnifiedVectorFormat &fmt, idx_t count, vector<float> &values, vector<uint8_t> &valid) {
	AppendMlxColumnT<T, float>(fmt, count, values, valid);
}

//! Buffers one column into the lane its type dictates: DECIMAL(<=18) raw into
//! int64 (exact lane), DATE as epoch days and other numerics into fp32.
static bool AppendLaneColumn(const LogicalType &type, UnifiedVectorFormat &fmt, idx_t count, vector<float> &values,
                             vector<int64_t> &ivalues, vector<uint8_t> &valid) {
	switch (type.id()) {
	case LogicalTypeId::DOUBLE:
		AppendMlxColumnT<double, float>(fmt, count, values, valid);
		return true;
	case LogicalTypeId::FLOAT:
		AppendMlxColumnT<float, float>(fmt, count, values, valid);
		return true;
	case LogicalTypeId::BIGINT:
		AppendMlxColumnT<int64_t, float>(fmt, count, values, valid);
		return true;
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::DATE: // physical int32 epoch days
		AppendMlxColumnT<int32_t, float>(fmt, count, values, valid);
		return true;
	case LogicalTypeId::DECIMAL:
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			AppendMlxColumnT<int16_t, int64_t>(fmt, count, ivalues, valid);
			return true;
		case PhysicalType::INT32:
			AppendMlxColumnT<int32_t, int64_t>(fmt, count, ivalues, valid);
			return true;
		case PhysicalType::INT64:
			AppendMlxColumnT<int64_t, int64_t>(fmt, count, ivalues, valid);
			return true;
		default:
			return false;
		}
	default:
		return false;
	}
}

//! After a filtered cold scan, warm the GPU cache from an unfiltered table read
//! so a follow-up MLX_SUM_CACHED can apply cache_filter over full columns.
static void WarmColumnCacheIfNeeded(ClientContext &context, const string &table_prefix, const vector<string> &col_keys,
                                    int64_t expected_rows) {
	if (col_keys.empty() || duckdb_mlx::MlxCacheHas(col_keys, expected_rows)) {
		return;
	}
	auto hash_pos = table_prefix.rfind('#');
	if (hash_pos == string::npos) {
		return;
	}
	auto qualified = table_prefix.substr(0, hash_pos);
	auto parts = StringUtil::Split(qualified, '.');
	if (parts.size() != 3) {
		return;
	}
	auto &catalog = Catalog::GetCatalog(context, parts[0]);
	CatalogTransaction catalog_transaction(catalog, context);
	auto &schema = catalog.GetSchema(context, parts[1]);
	auto &table = schema.GetEntry(catalog_transaction, CatalogType::TABLE_ENTRY, parts[2])->Cast<TableCatalogEntry>();

	vector<StorageIndex> storage_cols;
	vector<LogicalType> types;
	for (auto &key : col_keys) {
		auto pos = key.rfind('#');
		if (pos == string::npos) {
			return;
		}
		auto storage_id = std::stoul(key.substr(pos + 1));
		storage_cols.emplace_back(storage_id);
		types.push_back(table.GetColumn(LogicalIndex(storage_id)).Type());
	}

	auto &data_table = table.GetStorage();
	DuckTransaction &transaction = DuckTransaction::Get(context, table.catalog.GetAttached());
	TableScanState scan_state;
	data_table.InitializeScan(context, transaction, scan_state, storage_cols, nullptr);

	auto plan = duckdb_mlx::MlxCacheBeginPopulation(table_prefix, col_keys, expected_rows);
	DataChunk chunk;
	chunk.Initialize(context, types);

	// batch scan chunks into large segments — one segment per scan chunk
	// would mean thousands of tiny GPU graphs per hot query
	constexpr idx_t kWarmSegmentRows = 8 * 1024 * 1024;
	vector<vector<float>> values(types.size());
	vector<vector<int64_t>> ivalues(types.size());
	vector<vector<uint8_t>> valid(types.size());
	idx_t buffered = 0;

	auto flush = [&]() {
		if (buffered == 0) {
			return;
		}
		vector<MlxColumnData> cols;
		for (idx_t col = 0; col < values.size(); col++) {
			MlxColumnData data;
			data.valid = valid[col].empty() ? nullptr : valid[col].data();
			if (!ivalues[col].empty()) {
				data.ivalues = ivalues[col].data();
			} else {
				data.values = values[col].data();
			}
			cols.push_back(data);
		}
		duckdb_mlx::MlxCacheStoreSegment(plan.population, col_keys, plan.store_col, cols, buffered);
		for (idx_t col = 0; col < values.size(); col++) {
			values[col].clear();
			ivalues[col].clear();
			valid[col].clear();
		}
		buffered = 0;
	};

	while (true) {
		chunk.Reset();
		data_table.Scan(transaction, chunk, scan_state);
		auto count = chunk.size();
		if (count == 0) {
			break;
		}
		for (idx_t col = 0; col < chunk.ColumnCount(); col++) {
			UnifiedVectorFormat fmt;
			chunk.data[col].ToUnifiedFormat(count, fmt);
			if (!AppendLaneColumn(chunk.data[col].GetType(), fmt, count, values[col], ivalues[col], valid[col])) {
				return; // unsupported column type: leave the cache unwarmed
			}
		}
		buffered += count;
		if (buffered >= kWarmSegmentRows) {
			flush();
		}
	}
	flush();
	duckdb_mlx::LogDebug("MLX_SUM warmed the GPU cache (unfiltered) for " + table_prefix);
}

//===--------------------------------------------------------------------===//
// Physical operator: sinks child chunks into unified-memory column buffers,
// evaluates all SUM programs in one GPU graph at Finalize, emits one row.
//===--------------------------------------------------------------------===//
//! One thread's buffered slice of the input; segments are evaluated on the
//! GPU independently and their partial sums accumulated (SUM is decomposable),
//! so combining never re-copies the data.
struct MlxSumSegment {
	vector<vector<float>> values;
	vector<vector<int64_t>> ivalues; // int64 lane (DECIMAL raw)
	vector<vector<uint8_t>> valid;
};

static size_t SegmentRowCount(const MlxSumSegment &segment) {
	for (idx_t col = 0; col < segment.values.size(); col++) {
		if (!segment.values[col].empty()) {
			return segment.values[col].size();
		}
		if (!segment.ivalues[col].empty()) {
			return segment.ivalues[col].size();
		}
	}
	return 0;
}

static void SegmentColumns(const MlxSumSegment &segment, vector<duckdb_mlx::MlxColumnData> &cols) {
	for (idx_t col = 0; col < segment.values.size(); col++) {
		duckdb_mlx::MlxColumnData data;
		data.valid = segment.valid[col].empty() ? nullptr : segment.valid[col].data();
		if (!segment.ivalues[col].empty()) {
			data.ivalues = segment.ivalues[col].data();
		} else {
			data.values = segment.values[col].data();
		}
		cols.push_back(data);
	}
}

class MlxSumLocalSinkState : public LocalSinkState {
public:
	MlxSumSegment segment;
};

class MlxSumGlobalSinkState : public GlobalSinkState {
public:
	mutex glock;
	vector<MlxSumSegment> segments;
	vector<duckdb_mlx::MlxSumResult> results;
};

class MlxSumGlobalSourceState : public GlobalSourceState {
public:
	bool done = false;
};

class MlxSumPhysicalOperator : public PhysicalOperator {
public:
	MlxSumPhysicalOperator(PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality,
	                       vector<MlxSumProgram> programs_p, MlxFilter cache_filter_p, vector<string> col_keys_p,
	                       string table_prefix_p, int64_t expected_rows_p, bool cached_p, bool skip_cache_populate_p)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
	      programs(std::move(programs_p)), cache_filter(std::move(cache_filter_p)), col_keys(std::move(col_keys_p)),
	      table_prefix(std::move(table_prefix_p)), expected_rows(expected_rows_p), cached(cached_p),
	      skip_cache_populate(skip_cache_populate_p) {
	}

	vector<MlxSumProgram> programs;
	//! GPU row mask for MLX_SUM_CACHED only (full-table cache + pushed filters).
	MlxFilter cache_filter;
	//! GPU cache keys of the child's output columns; empty when caching is off
	vector<string> col_keys;
	string table_prefix;
	int64_t expected_rows = 0;
	//! When true the operator has no child: it is a pure source reading the
	//! GPU-resident column cache (no table scan at all)
	bool cached;
	//! Cold scans with pushed-down table filters skip cache population (Sirius:
	//! scan applies filters; cache holds unfiltered columns from other queries).
	bool skip_cache_populate;

	string GetName() const override {
		return cached ? "MLX_SUM_CACHED" : "MLX_SUM";
	}

	// Sink interface
	bool IsSink() const override {
		return !cached;
	}
	bool ParallelSink() const override {
		return true;
	}

	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override {
		return make_uniq<MlxSumLocalSinkState>();
	}
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		return make_uniq<MlxSumGlobalSinkState>();
	}

	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override {
		auto &segment = input.local_state.Cast<MlxSumLocalSinkState>().segment;
		if (segment.values.empty()) {
			segment.values.resize(chunk.ColumnCount());
			segment.ivalues.resize(chunk.ColumnCount());
			segment.valid.resize(chunk.ColumnCount()); // stays empty per column until a NULL appears
		}
		auto count = chunk.size();
		for (idx_t col = 0; col < chunk.ColumnCount(); col++) {
			UnifiedVectorFormat fmt;
			chunk.data[col].ToUnifiedFormat(count, fmt);
			if (!AppendLaneColumn(chunk.data[col].GetType(), fmt, count, segment.values[col], segment.ivalues[col],
			                      segment.valid[col])) {
				throw InternalException("MLX_SUM: unexpected column type at execution time");
			}
		}
		return SinkResultType::NEED_MORE_INPUT;
	}

	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override {
		auto &lstate = input.local_state.Cast<MlxSumLocalSinkState>();
		auto &gstate = input.global_state.Cast<MlxSumGlobalSinkState>();
		if (lstate.segment.values.empty()) {
			return SinkCombineResultType::FINISHED;
		}
		lock_guard<mutex> guard(gstate.glock);
		gstate.segments.push_back(std::move(lstate.segment));
		return SinkCombineResultType::FINISHED;
	}

	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override {
		auto &gstate = input.global_state.Cast<MlxSumGlobalSinkState>();
		if (!col_keys.empty() && !skip_cache_populate) {
			// populate the GPU column cache (incremental: reuse columns already
			// resident), then evaluate from the cache
			auto plan = duckdb_mlx::MlxCacheBeginPopulation(table_prefix, col_keys, expected_rows);
			for (auto &segment : gstate.segments) {
				vector<MlxColumnData> cols;
				size_t row_count = SegmentRowCount(segment);
				SegmentColumns(segment, cols);
				duckdb_mlx::MlxCacheStoreSegment(plan.population, col_keys, plan.store_col, cols, row_count);
			}
			duckdb_mlx::MlxFilter no_filter;
			gstate.results = duckdb_mlx::MlxSumExprsCached(col_keys, programs, no_filter);
			duckdb_mlx::LogDebug("MLX_SUM populated the GPU cache for " + table_prefix);
			return SinkFinalizeType::READY;
		}
		gstate.results.clear();
		for (auto &program : programs) {
			switch (program.kind) {
			case MlxAggKind::MIN:
				gstate.results.push_back(
				    {std::numeric_limits<double>::infinity(), 0, std::numeric_limits<int64_t>::max()});
				break;
			case MlxAggKind::MAX:
				gstate.results.push_back(
				    {-std::numeric_limits<double>::infinity(), 0, std::numeric_limits<int64_t>::min()});
				break;
			default:
				gstate.results.push_back({0.0, 0, 0});
				break;
			}
		}
		duckdb_mlx::MlxFilter no_filter;
		for (auto &segment : gstate.segments) {
			vector<MlxColumnData> cols;
			size_t row_count = SegmentRowCount(segment);
			SegmentColumns(segment, cols);
			auto partial = duckdb_mlx::MlxSumExprs(cols, row_count, programs, no_filter);
			for (idx_t i = 0; i < programs.size(); i++) {
				switch (programs[i].kind) {
				case MlxAggKind::MIN:
					gstate.results[i].value = MinValue(gstate.results[i].value, partial[i].value);
					gstate.results[i].ivalue = MinValue(gstate.results[i].ivalue, partial[i].ivalue);
					break;
				case MlxAggKind::MAX:
					gstate.results[i].value = MaxValue(gstate.results[i].value, partial[i].value);
					gstate.results[i].ivalue = MaxValue(gstate.results[i].ivalue, partial[i].ivalue);
					break;
				case MlxAggKind::AVG:
					// re-weight the per-segment average into a running sum;
					// int-lane averages accumulate exactly through ivalue
					gstate.results[i].value += partial[i].value * static_cast<double>(partial[i].valid_count);
					gstate.results[i].ivalue += partial[i].ivalue;
					break;
				default:
					gstate.results[i].value += partial[i].value;
					gstate.results[i].ivalue += partial[i].ivalue;
					break;
				}
				gstate.results[i].valid_count += partial[i].valid_count;
			}
		}
		for (idx_t i = 0; i < programs.size(); i++) {
			if (programs[i].kind == MlxAggKind::AVG && gstate.results[i].valid_count > 0) {
				auto count = static_cast<double>(gstate.results[i].valid_count);
				gstate.results[i].value = programs[i].int_lane ? static_cast<double>(gstate.results[i].ivalue) / count
				                                               : gstate.results[i].value / count;
			}
		}
		if (skip_cache_populate) {
			WarmColumnCacheIfNeeded(context, table_prefix, col_keys, expected_rows);
		}
		return SinkFinalizeType::READY;
	}

	// Source interface
	bool IsSource() const override {
		return true;
	}
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override {
		return make_uniq<MlxSumGlobalSourceState>();
	}

	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override {
		auto &source_state = input.global_state.Cast<MlxSumGlobalSourceState>();
		if (source_state.done) {
			return SourceResultType::FINISHED;
		}
		vector<duckdb_mlx::MlxSumResult> results;
		if (cached) {
			// pure source: evaluate straight from the GPU-resident cache
			results = duckdb_mlx::MlxSumExprsCached(col_keys, programs, cache_filter);
		} else {
			results = sink_state->Cast<MlxSumGlobalSinkState>().results;
		}
		for (idx_t i = 0; i < results.size(); i++) {
			if (programs[i].kind == MlxAggKind::COUNT || programs[i].kind == MlxAggKind::COUNT_STAR) {
				FlatVector::GetData<int64_t>(chunk.data[i])[0] = results[i].valid_count;
			} else if (results[i].valid_count == 0) {
				FlatVector::Validity(chunk.data[i]).SetInvalid(0);
			} else if (programs[i].int_lane && types[i].id() == LogicalTypeId::DECIMAL) {
				// exact raw-scaled integer result
				switch (types[i].InternalType()) {
				case PhysicalType::INT128:
					FlatVector::GetData<hugeint_t>(chunk.data[i])[0] = hugeint_t(results[i].ivalue);
					break;
				case PhysicalType::INT64:
					FlatVector::GetData<int64_t>(chunk.data[i])[0] = results[i].ivalue;
					break;
				case PhysicalType::INT32:
					FlatVector::GetData<int32_t>(chunk.data[i])[0] = NumericCast<int32_t>(results[i].ivalue);
					break;
				default:
					FlatVector::GetData<int16_t>(chunk.data[i])[0] = NumericCast<int16_t>(results[i].ivalue);
					break;
				}
			} else {
				FlatVector::GetData<double>(chunk.data[i])[0] = results[i].value * programs[i].render_scale;
			}
		}
		chunk.SetCardinality(1);
		source_state.done = true;
		return SourceResultType::FINISHED;
	}
};

//===--------------------------------------------------------------------===//
// Logical operator
//===--------------------------------------------------------------------===//
class MlxSumLogicalOperator : public LogicalExtensionOperator {
public:
	MlxSumLogicalOperator(idx_t aggregate_index, vector<MlxSumProgram> programs_p, vector<LogicalType> agg_types_p,
	                      MlxFilter cache_filter_p, vector<string> col_keys_p, string table_prefix_p,
	                      int64_t expected_rows_p, bool cached_p, bool skip_cache_populate_p)
	    : aggregate_index(aggregate_index), programs(std::move(programs_p)), agg_types(std::move(agg_types_p)),
	      cache_filter(std::move(cache_filter_p)), col_keys(std::move(col_keys_p)),
	      table_prefix(std::move(table_prefix_p)), expected_rows(expected_rows_p), cached(cached_p),
	      skip_cache_populate(skip_cache_populate_p) {
		estimated_cardinality = 1;
		has_estimated_cardinality = true;
	}

	idx_t aggregate_index;
	vector<MlxSumProgram> programs;
	vector<LogicalType> agg_types;
	MlxFilter cache_filter;
	vector<string> col_keys;
	string table_prefix;
	int64_t expected_rows = 0;
	bool cached;
	bool skip_cache_populate;

	string GetName() const override {
		return cached ? "MLX_SUM_CACHED" : "MLX_SUM";
	}
	string GetExtensionName() const override {
		return "duckdb_mlx";
	}

	vector<ColumnBinding> GetColumnBindings() override {
		vector<ColumnBinding> bindings;
		for (idx_t i = 0; i < programs.size(); i++) {
			bindings.emplace_back(aggregate_index, i);
		}
		return bindings;
	}

	void ResolveTypes() override {
		types = agg_types;
	}

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override {
		auto &op = planner.Make<MlxSumPhysicalOperator>(
		    types, estimated_cardinality, std::move(programs), std::move(cache_filter), std::move(col_keys),
		    std::move(table_prefix), expected_rows, cached, skip_cache_populate);
		if (!cached) {
			auto &child = planner.CreatePlan(*children[0]);
			op.children.push_back(child);
		}
		return op;
	}
};

//===--------------------------------------------------------------------===//
// GROUP BY: single integer key + single SUM(column) on GPU (hash or sort).
//===--------------------------------------------------------------------===//
struct MlxGroupbySegment {
	vector<int64_t> keys;
	vector<double> values;
};

class MlxGroupbyLocalSinkState : public LocalSinkState {
public:
	MlxGroupbySegment segment;
};

class MlxGroupbyGlobalSinkState : public GlobalSinkState {
public:
	mutex glock;
	vector<MlxGroupbySegment> segments;
	vector<duckdb_mlx::MlxGroupbyRow> results;
};

class MlxGroupbyGlobalSourceState : public GlobalSourceState {
public:
	idx_t offset = 0;
	//! results for the cached (scan-less) mode, computed once per execution
	vector<duckdb_mlx::MlxGroupbyRow> cached_rows;
	bool cached_ready = false;
};

class MlxGroupbyPhysicalOperator : public PhysicalOperator {
public:
	MlxGroupbyPhysicalOperator(PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality,
	                           int32_t group_col_p, int32_t value_col_p, vector<string> col_keys_p,
	                           string table_prefix_p, int64_t expected_rows_p, bool cached_p)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
	      group_col(group_col_p), value_col(value_col_p), col_keys(std::move(col_keys_p)),
	      table_prefix(std::move(table_prefix_p)), expected_rows(expected_rows_p), cached(cached_p) {
	}

	int32_t group_col;
	int32_t value_col;
	vector<string> col_keys;
	string table_prefix;
	int64_t expected_rows = 0;
	bool cached;

	string GetName() const override {
		return cached ? "MLX_GROUPBY_CACHED" : "MLX_GROUPBY";
	}

	bool IsSink() const override {
		return !cached;
	}
	bool ParallelSink() const override {
		return true;
	}
	bool IsSource() const override {
		return true;
	}

	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override {
		return make_uniq<MlxGroupbyLocalSinkState>();
	}
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		return make_uniq<MlxGroupbyGlobalSinkState>();
	}
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override {
		return make_uniq<MlxGroupbyGlobalSourceState>();
	}

	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override {
		auto &segment = input.local_state.Cast<MlxGroupbyLocalSinkState>().segment;
		auto count = chunk.size();
		auto base = segment.keys.size();
		segment.keys.resize(base + count);
		segment.values.resize(base + count);

		UnifiedVectorFormat gfmt;
		chunk.data[group_col].ToUnifiedFormat(count, gfmt);
		UnifiedVectorFormat vfmt;
		chunk.data[value_col].ToUnifiedFormat(count, vfmt);

		switch (chunk.data[group_col].GetType().id()) {
		case LogicalTypeId::BIGINT: {
			auto gdata = UnifiedVectorFormat::GetData<int64_t>(gfmt);
			for (idx_t i = 0; i < count; i++) {
				auto idx = gfmt.sel->get_index(i);
				segment.keys[base + i] = gfmt.validity.RowIsValid(idx) ? gdata[idx] : INT64_MIN;
			}
			break;
		}
		case LogicalTypeId::INTEGER: {
			auto gdata = UnifiedVectorFormat::GetData<int32_t>(gfmt);
			for (idx_t i = 0; i < count; i++) {
				auto idx = gfmt.sel->get_index(i);
				segment.keys[base + i] = gfmt.validity.RowIsValid(idx) ? gdata[idx] : INT64_MIN;
			}
			break;
		}
		case LogicalTypeId::UTINYINT: {
			auto gdata = UnifiedVectorFormat::GetData<uint8_t>(gfmt);
			for (idx_t i = 0; i < count; i++) {
				auto idx = gfmt.sel->get_index(i);
				segment.keys[base + i] = gfmt.validity.RowIsValid(idx) ? gdata[idx] : INT64_MIN;
			}
			break;
		}
		case LogicalTypeId::USMALLINT: {
			auto gdata = UnifiedVectorFormat::GetData<uint16_t>(gfmt);
			for (idx_t i = 0; i < count; i++) {
				auto idx = gfmt.sel->get_index(i);
				segment.keys[base + i] = gfmt.validity.RowIsValid(idx) ? gdata[idx] : INT64_MIN;
			}
			break;
		}
		case LogicalTypeId::UINTEGER: {
			auto gdata = UnifiedVectorFormat::GetData<uint32_t>(gfmt);
			for (idx_t i = 0; i < count; i++) {
				auto idx = gfmt.sel->get_index(i);
				segment.keys[base + i] = gfmt.validity.RowIsValid(idx) ? gdata[idx] : INT64_MIN;
			}
			break;
		}
		default:
			throw InternalException("MLX_GROUPBY: unsupported group key type");
		}

		switch (chunk.data[value_col].GetType().id()) {
		case LogicalTypeId::DOUBLE: {
			auto vdata = UnifiedVectorFormat::GetData<double>(vfmt);
			for (idx_t i = 0; i < count; i++) {
				auto idx = vfmt.sel->get_index(i);
				segment.values[base + i] = vfmt.validity.RowIsValid(idx) ? vdata[idx] : 0.0;
			}
			break;
		}
		case LogicalTypeId::FLOAT: {
			auto vdata = UnifiedVectorFormat::GetData<float>(vfmt);
			for (idx_t i = 0; i < count; i++) {
				auto idx = vfmt.sel->get_index(i);
				segment.values[base + i] = vfmt.validity.RowIsValid(idx) ? static_cast<double>(vdata[idx]) : 0.0;
			}
			break;
		}
		case LogicalTypeId::BIGINT: {
			auto vdata = UnifiedVectorFormat::GetData<int64_t>(vfmt);
			for (idx_t i = 0; i < count; i++) {
				auto idx = vfmt.sel->get_index(i);
				segment.values[base + i] = vfmt.validity.RowIsValid(idx) ? static_cast<double>(vdata[idx]) : 0.0;
			}
			break;
		}
		case LogicalTypeId::INTEGER: {
			auto vdata = UnifiedVectorFormat::GetData<int32_t>(vfmt);
			for (idx_t i = 0; i < count; i++) {
				auto idx = vfmt.sel->get_index(i);
				segment.values[base + i] = vfmt.validity.RowIsValid(idx) ? static_cast<double>(vdata[idx]) : 0.0;
			}
			break;
		}
		default:
			throw InternalException("MLX_GROUPBY: unsupported value type");
		}
		return SinkResultType::NEED_MORE_INPUT;
	}

	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override {
		auto &lstate = input.local_state.Cast<MlxGroupbyLocalSinkState>();
		auto &gstate = input.global_state.Cast<MlxGroupbyGlobalSinkState>();
		if (lstate.segment.keys.empty()) {
			return SinkCombineResultType::FINISHED;
		}
		lock_guard<mutex> guard(gstate.glock);
		gstate.segments.push_back(std::move(lstate.segment));
		return SinkCombineResultType::FINISHED;
	}

	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override {
		auto &gstate = input.global_state.Cast<MlxGroupbyGlobalSinkState>();
		if (col_keys.size() == 2) {
			auto plan = duckdb_mlx::MlxCacheBeginPopulation(table_prefix, col_keys, expected_rows);
			for (auto &segment : gstate.segments) {
				vector<float> key_f(segment.keys.begin(), segment.keys.end());
				vector<float> val_f(segment.values.begin(), segment.values.end());
				vector<duckdb_mlx::MlxColumnData> cols = {{key_f.data(), nullptr}, {val_f.data(), nullptr}};
				duckdb_mlx::MlxCacheStoreSegment(plan.population, col_keys, plan.store_col, cols, key_f.size());
				duckdb_mlx::MlxGroupbyDenseAccumulateHost(col_keys[0], col_keys[1], plan.population, key_f.data(),
				                                          val_f.data(), key_f.size());
			}
			vector<int64_t> keys;
			vector<double> values;
			for (auto &segment : gstate.segments) {
				keys.insert(keys.end(), segment.keys.begin(), segment.keys.end());
				values.insert(values.end(), segment.values.begin(), segment.values.end());
			}
			gstate.results = duckdb_mlx::MlxGroupbySum(keys.data(), values.data(), nullptr, keys.size(), false);
			return SinkFinalizeType::READY;
		}
		vector<int64_t> keys;
		vector<double> values;
		for (auto &segment : gstate.segments) {
			keys.insert(keys.end(), segment.keys.begin(), segment.keys.end());
			values.insert(values.end(), segment.values.begin(), segment.values.end());
		}
		gstate.results = duckdb_mlx::MlxGroupbySum(keys.data(), values.data(), nullptr, keys.size(), false);
		return SinkFinalizeType::READY;
	}

	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override {
		auto &source_state = input.global_state.Cast<MlxGroupbyGlobalSourceState>();
		if (cached && !source_state.cached_ready) {
			// pure source: never touches sink_state (there is none)
			source_state.cached_rows = duckdb_mlx::MlxGroupbySumCached(col_keys[0], col_keys[1]);
			source_state.cached_ready = true;
		}
		auto &results = cached ? source_state.cached_rows : sink_state->Cast<MlxGroupbyGlobalSinkState>().results;
		if (types.size() != 2) {
			throw InternalException("MLX_GROUPBY: operator types.size=" + std::to_string(types.size()));
		}
		if (results.empty() || source_state.offset >= results.size()) {
			return SourceResultType::FINISHED;
		}
		auto remaining = results.size() - source_state.offset;
		auto emit = MinValue<idx_t>(remaining, STANDARD_VECTOR_SIZE);
		switch (chunk.data[0].GetType().id()) {
		case LogicalTypeId::BIGINT: {
			auto key_data = FlatVector::GetData<int64_t>(chunk.data[0]);
			for (idx_t i = 0; i < emit; i++) {
				auto &row = results[source_state.offset + i];
				key_data[i] = row.key;
			}
			break;
		}
		case LogicalTypeId::INTEGER: {
			auto key_data = FlatVector::GetData<int32_t>(chunk.data[0]);
			for (idx_t i = 0; i < emit; i++) {
				auto &row = results[source_state.offset + i];
				key_data[i] = static_cast<int32_t>(row.key);
			}
			break;
		}
		case LogicalTypeId::UTINYINT: {
			auto key_data = FlatVector::GetData<uint8_t>(chunk.data[0]);
			for (idx_t i = 0; i < emit; i++) {
				auto &row = results[source_state.offset + i];
				key_data[i] = static_cast<uint8_t>(row.key);
			}
			break;
		}
		case LogicalTypeId::USMALLINT: {
			auto key_data = FlatVector::GetData<uint16_t>(chunk.data[0]);
			for (idx_t i = 0; i < emit; i++) {
				auto &row = results[source_state.offset + i];
				key_data[i] = static_cast<uint16_t>(row.key);
			}
			break;
		}
		case LogicalTypeId::UINTEGER: {
			auto key_data = FlatVector::GetData<uint32_t>(chunk.data[0]);
			for (idx_t i = 0; i < emit; i++) {
				auto &row = results[source_state.offset + i];
				key_data[i] = static_cast<uint32_t>(row.key);
			}
			break;
		}
		default:
			throw InternalException("MLX_GROUPBY: unsupported output group key type");
		}
		auto sum_data = FlatVector::GetData<double>(chunk.data[1]);
		for (idx_t i = 0; i < emit; i++) {
			sum_data[i] = results[source_state.offset + i].sum;
		}
		chunk.SetCardinality(emit);
		source_state.offset += emit;
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
};

class MlxGroupbyLogicalOperator : public LogicalExtensionOperator {
public:
	MlxGroupbyLogicalOperator(idx_t group_index_p, idx_t aggregate_index_p, int32_t group_col_p, int32_t value_col_p,
	                          vector<LogicalType> output_types_p, vector<string> col_keys_p, string table_prefix_p,
	                          int64_t expected_rows_p, bool cached_p)
	    : group_index(group_index_p), aggregate_index(aggregate_index_p), group_col(group_col_p),
	      value_col(value_col_p), output_types(std::move(output_types_p)), col_keys(std::move(col_keys_p)),
	      table_prefix(std::move(table_prefix_p)), expected_rows(expected_rows_p), cached(cached_p) {
	}

	idx_t group_index;
	idx_t aggregate_index;
	int32_t group_col;
	int32_t value_col;
	vector<LogicalType> output_types;
	vector<string> col_keys;
	string table_prefix;
	int64_t expected_rows = 0;
	bool cached;

	string GetName() const override {
		return cached ? "MLX_GROUPBY_CACHED" : "MLX_GROUPBY";
	}
	string GetExtensionName() const override {
		return "duckdb_mlx";
	}

	vector<ColumnBinding> GetColumnBindings() override {
		return {{group_index, 0}, {aggregate_index, 0}};
	}

	void ResolveTypes() override {
		types = output_types;
	}

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override {
		auto &op = planner.Make<MlxGroupbyPhysicalOperator>(output_types, estimated_cardinality, group_col, value_col,
		                                                    col_keys, table_prefix, expected_rows, cached);
		if (!cached) {
			auto &child = planner.CreatePlan(*children[0]);
			op.children.push_back(child);
		}
		return op;
	}
};

//===--------------------------------------------------------------------===//
// Cost model: decline GPU when DuckDB CPU is already optimal (§1.1 PLAN).
//===--------------------------------------------------------------------===//

static bool IsPlainColumnProgram(const duckdb_mlx::MlxSumProgram &program) {
	if (program.kind == duckdb_mlx::MlxAggKind::COUNT_STAR) {
		return true;
	}
	if (program.ops.size() != 1 || program.ops[0].code != duckdb_mlx::MlxExprOpCode::LOAD_COL) {
		return false;
	}
	switch (program.kind) {
	case duckdb_mlx::MlxAggKind::COUNT:
	case duckdb_mlx::MlxAggKind::SUM:
	case duckdb_mlx::MlxAggKind::AVG:
	case duckdb_mlx::MlxAggKind::MIN:
	case duckdb_mlx::MlxAggKind::MAX:
		return true;
	default:
		return false;
	}
}

static bool ProgramsAreCpuFast(const vector<duckdb_mlx::MlxSumProgram> &programs) {
	if (programs.size() != 1) {
		return false;
	}
	return IsPlainColumnProgram(programs[0]);
}

//===--------------------------------------------------------------------===//
// Optimizer hook: match AGGREGATE <- [PROJECTION] <- [FILTER] <- GET and
// replace. Sirius-style: DuckDB scan applies table_filters; residual FILTER
// stays in the plan on the cold path. GPU masks (cache_filter) apply only on
// MLX_SUM_CACHED reads over the resident column cache.
//===--------------------------------------------------------------------===//

static void CollectProgramColumnIds(const vector<MlxSumProgram> &programs, const LogicalGet &get,
                                    std::set<idx_t> &column_ids_out) {
	auto &column_ids = get.GetColumnIds();
	auto storage_to_column_ids_idx = [&](idx_t storage_col) -> idx_t {
		for (idx_t i = 0; i < column_ids.size(); i++) {
			if (column_ids[i].GetPrimaryIndex() == storage_col) {
				return i;
			}
		}
		return column_ids.size();
	};
	for (auto &program : programs) {
		for (auto storage_col : program.null_cols) {
			auto column_ids_idx = storage_to_column_ids_idx(static_cast<idx_t>(storage_col));
			if (column_ids_idx < column_ids.size()) {
				column_ids_out.insert(column_ids_idx);
			}
		}
		for (auto &op : program.ops) {
			if (op.code != MlxExprOpCode::LOAD_COL) {
				continue;
			}
			auto column_ids_idx = storage_to_column_ids_idx(static_cast<idx_t>(op.col));
			if (column_ids_idx < column_ids.size()) {
				column_ids_out.insert(column_ids_idx);
			}
		}
	}
}

static bool RemapStorageToChunk(std::vector<MlxExprOp> &ops, std::vector<int32_t> &null_cols,
                                const vector<idx_t> &chunk_projection, const LogicalGet &get) {
	auto &column_ids = get.GetColumnIds();
	auto storage_to_column_ids_idx = [&](idx_t storage_col) -> idx_t {
		for (idx_t i = 0; i < column_ids.size(); i++) {
			if (column_ids[i].GetPrimaryIndex() == storage_col) {
				return i;
			}
		}
		return column_ids.size();
	};
	auto storage_to_chunk = [&](int32_t storage_col) -> int32_t {
		auto column_ids_idx = storage_to_column_ids_idx(static_cast<idx_t>(storage_col));
		if (column_ids_idx >= column_ids.size()) {
			return -1;
		}
		for (idx_t out = 0; out < chunk_projection.size(); out++) {
			if (chunk_projection[out] == column_ids_idx) {
				return NumericCast<int32_t>(out);
			}
		}
		return -1;
	};
	for (auto &op : ops) {
		if (op.code == MlxExprOpCode::LOAD_COL) {
			auto chunk_col = storage_to_chunk(op.col);
			if (chunk_col < 0) {
				return false;
			}
			op.col = chunk_col;
		}
	}
	for (auto &col : null_cols) {
		auto chunk_col = storage_to_chunk(col);
		if (chunk_col < 0) {
			return false;
		}
		col = chunk_col;
	}
	return true;
}

//! Translates a pushed-down TableFilter tree into predicate IR. `col_pos` is
//! the column's position in the GET's column_ids (the space LOAD_COL uses
//! before remapping); `storage_idx` indexes the table schema for type checks.
static bool TranslateTableFilter(const TableFilter &table_filter, idx_t col_pos, idx_t storage_idx,
                                 const LogicalGet &get, vector<MlxExprOp> &ops, std::set<int32_t> &referenced) {
	switch (table_filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = table_filter.Cast<ConstantFilter>();
		if (constant_filter.constant.IsNull()) {
			return false;
		}
		MlxExprOpCode code;
		switch (constant_filter.comparison_type) {
		case ExpressionType::COMPARE_LESSTHAN:
			code = MlxExprOpCode::CMP_LT;
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			code = MlxExprOpCode::CMP_LE;
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
			code = MlxExprOpCode::CMP_GT;
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			code = MlxExprOpCode::CMP_GE;
			break;
		case ExpressionType::COMPARE_EQUAL:
			code = MlxExprOpCode::CMP_EQ;
			break;
		case ExpressionType::COMPARE_NOTEQUAL:
			code = MlxExprOpCode::CMP_NE;
			break;
		default:
			return false;
		}
		if (storage_idx >= get.returned_types.size() ||
		    !MlxExprTranslator::IsSupportedColumnType(get.returned_types[storage_idx])) {
			return false;
		}
		auto &col_type = get.returned_types[storage_idx];
		Value konst = constant_filter.constant;
		MlxExprOp const_op {MlxExprOpCode::CONST_VAL, 0, 0};
		if (MlxExprTranslator::IsIntLaneType(col_type)) {
			if (konst.type() != col_type && !konst.DefaultTryCastAs(col_type)) {
				duckdb_mlx::LogDebug("MLX: decimal filter constant of type " + konst.type().ToString() + " declined");
				return false;
			}
			const_op.ivalue = MlxExprTranslator::DecimalRaw(konst);
			const_op.int_lane = true;
		} else if (col_type.id() == LogicalTypeId::DATE) {
			if (konst.type().id() != LogicalTypeId::DATE && !konst.DefaultTryCastAs(LogicalType::DATE)) {
				duckdb_mlx::LogDebug("MLX: date filter constant of type " + konst.type().ToString() + " declined");
				return false;
			}
			const_op.value = static_cast<double>(konst.GetValue<date_t>().days);
		} else if (konst.type().IsNumeric()) {
			const_op.value = konst.GetValue<double>();
		} else {
			duckdb_mlx::LogDebug("MLX: filter constant of type " + konst.type().ToString() + " declined");
			return false;
		}
		ops.push_back({MlxExprOpCode::LOAD_COL, NumericCast<int32_t>(storage_idx), 0});
		ops.push_back(const_op);
		ops.push_back({code, 0, 0});
		referenced.insert(NumericCast<int32_t>(storage_idx));
		return true;
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjunction = table_filter.Cast<ConjunctionAndFilter>();
		bool first = true;
		for (auto &child : conjunction.child_filters) {
			if (!TranslateTableFilter(*child, col_pos, storage_idx, get, ops, referenced)) {
				return false;
			}
			if (!first) {
				ops.push_back({MlxExprOpCode::AND, 0, 0});
			}
			first = false;
		}
		return !first;
	}
	case TableFilterType::CONJUNCTION_OR: {
		auto &conjunction = table_filter.Cast<ConjunctionOrFilter>();
		bool first = true;
		for (auto &child : conjunction.child_filters) {
			if (!TranslateTableFilter(*child, col_pos, storage_idx, get, ops, referenced)) {
				return false;
			}
			if (!first) {
				ops.push_back({MlxExprOpCode::OR, 0, 0});
			}
			first = false;
		}
		return !first;
	}
	case TableFilterType::IS_NOT_NULL:
		// handled purely through the column's validity mask
		referenced.insert(NumericCast<int32_t>(storage_idx));
		return true;
	default:
		duckdb_mlx::LogDebug("MLX: unsupported table filter type " +
		                     std::to_string(static_cast<int>(table_filter.filter_type)) + " on column " +
		                     std::to_string(storage_idx));
		return false;
	}
}

static bool TryInterceptGroupBy(ClientContext &context, unique_ptr<LogicalOperator> &plan, idx_t min_rows) {
	if (plan->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return false;
	}
	auto &agg = plan->Cast<LogicalAggregate>();
	if (agg.groups.size() != 1) {
		return false;
	}
	if (agg.grouping_sets.size() > 1) {
		return false;
	}
	if (!agg.grouping_sets.empty() && agg.grouping_sets[0].size() != 1) {
		return false;
	}
	if (agg.expressions.size() != 1) {
		return false;
	}
	if (agg.expressions[0]->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
		return false;
	}
	auto &aggr = agg.expressions[0]->Cast<BoundAggregateExpression>();
	if (aggr.function.name != "sum" || aggr.IsDistinct() || aggr.filter || aggr.children.size() != 1) {
		return false;
	}

	optional_ptr<LogicalProjection> child_proj;
	if (agg.children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		child_proj = &agg.children[0]->Cast<LogicalProjection>();
	}
	optional_ptr<LogicalFilter> filter_op;
	optional_ptr<LogicalGet> get;
	reference<LogicalOperator> node = *agg.children[0];
	while (node.get().type == LogicalOperatorType::LOGICAL_PROJECTION) {
		node = *node.get().children[0];
	}
	if (node.get().type == LogicalOperatorType::LOGICAL_FILTER) {
		filter_op = &node.get().Cast<LogicalFilter>();
		if (!filter_op->projection_map.empty()) {
			return false;
		}
		node = *filter_op->children[0];
	}
	if (node.get().type != LogicalOperatorType::LOGICAL_GET) {
		return false;
	}
	get = &node.get().Cast<LogicalGet>();
	if (get->function.name != "seq_scan") {
		return false;
	}
	auto estimated_rows =
	    get->has_estimated_cardinality ? get->estimated_cardinality : get->EstimateCardinality(context);
	if (estimated_rows < min_rows) {
		return false;
	}

	auto map_col = [&](const BoundColumnRefExpression &colref) -> int32_t {
		const BoundColumnRefExpression *cref = &colref;
		if (cref->binding.table_index == agg.group_index) {
			if (agg.groups[cref->binding.column_index]->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
				cref = &agg.groups[cref->binding.column_index]->Cast<BoundColumnRefExpression>();
			}
		}
		if (child_proj && cref->binding.table_index == child_proj->table_index) {
			auto &pexpr = child_proj->expressions[cref->binding.column_index];
			if (pexpr->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				return -1;
			}
			cref = &pexpr->Cast<BoundColumnRefExpression>();
		}
		if (cref->binding.table_index != get->table_index) {
			return -1;
		}
		auto col_idx = cref->binding.column_index;
		if (get->projection_ids.empty()) {
			return NumericCast<int32_t>(col_idx);
		}
		for (idx_t out = 0; out < get->projection_ids.size(); out++) {
			if (get->projection_ids[out] == col_idx) {
				return NumericCast<int32_t>(out);
			}
		}
		return -1;
	};

	auto resolve_output_idx = [&](const Expression &expr) -> int32_t {
		switch (expr.GetExpressionClass()) {
		case ExpressionClass::BOUND_COLUMN_REF: {
			auto &cref = expr.Cast<BoundColumnRefExpression>();
			if (child_proj && cref.binding.table_index == child_proj->table_index) {
				return NumericCast<int32_t>(cref.binding.column_index);
			}
			return map_col(cref);
		}
		case ExpressionClass::BOUND_REF:
			return NumericCast<int32_t>(expr.Cast<BoundReferenceExpression>().index);
		default:
			return -1;
		}
	};

	auto group_col = resolve_output_idx(*agg.groups[0]);
	auto value_col = resolve_output_idx(*aggr.children[0]);
	if (group_col < 0 || value_col < 0) {
		return false;
	}

	// NULL group keys / values have SQL semantics (NULL group, NULL-skipping
	// sums) the sink's sentinel encoding does not implement — decline unless
	// statistics prove both columns NULL-free
	auto column_may_be_null = [&](int32_t scan_col) {
		idx_t storage_idx = static_cast<idx_t>(scan_col);
		if (!get->projection_ids.empty()) {
			if (static_cast<size_t>(scan_col) >= get->projection_ids.size()) {
				return true;
			}
			storage_idx = get->projection_ids[scan_col];
		}
		if (storage_idx >= get->GetColumnIds().size()) {
			return true;
		}
		unique_ptr<BaseStatistics> stats;
		if (get->function.statistics_extended) {
			TableFunctionGetStatisticsInput input(get->bind_data.get(), get->GetColumnIds()[storage_idx]);
			stats = get->function.statistics_extended(context, input);
		} else if (get->function.statistics) {
			stats = get->function.statistics(context, get->bind_data.get(),
			                                 get->GetColumnIds()[storage_idx].GetPrimaryIndex());
		}
		return !stats || stats->CanHaveNull();
	};
	if (column_may_be_null(group_col) || column_may_be_null(value_col)) {
		return false;
	}
	LogicalType group_type;
	if (child_proj && group_col >= 0 && static_cast<size_t>(group_col) < child_proj->expressions.size()) {
		group_type = child_proj->expressions[group_col]->return_type;
	} else {
		group_type = agg.groups[0]->return_type;
	}
	if (group_type.id() != LogicalTypeId::BIGINT && group_type.id() != LogicalTypeId::INTEGER &&
	    group_type.id() != LogicalTypeId::UTINYINT && group_type.id() != LogicalTypeId::USMALLINT &&
	    group_type.id() != LogicalTypeId::UINTEGER) {
		return false;
	}
	if (!MlxExprTranslator::IsSupportedColumnType(aggr.children[0]->return_type)) {
		return false;
	}

	bool has_filter = filter_op != nullptr || !get->table_filters.filters.empty();

	vector<LogicalType> output_types = {group_type, aggr.return_type};

	vector<string> col_keys;
	string table_prefix;
	int64_t total_rows = 0;
	bool cached = false;
	auto table = get->GetTable();
	if (table) {
		table_prefix = table->ParentCatalog().GetName() + "." + table->ParentSchema().name + "." + table->name + "#";
		total_rows = NumericCast<int64_t>(table->GetStorage().GetTotalRows());
		auto storage_key_for_scan_col = [&](int32_t scan_col) -> string {
			if (scan_col < 0) {
				return "";
			}
			idx_t storage_idx = static_cast<idx_t>(scan_col);
			if (!get->projection_ids.empty()) {
				if (static_cast<size_t>(scan_col) >= get->projection_ids.size()) {
					return "";
				}
				storage_idx = get->projection_ids[scan_col];
			}
			if (storage_idx >= get->GetColumnIds().size()) {
				return "";
			}
			return table_prefix + std::to_string(get->GetColumnIds()[storage_idx].GetPrimaryIndex());
		};
		auto group_key = storage_key_for_scan_col(group_col);
		auto value_key = storage_key_for_scan_col(value_col);
		if (!group_key.empty() && !value_key.empty() && !has_filter) {
			col_keys = {group_key, value_key};
			// serving from the cache requires provable correctness: dense
			// table ready, or fp32-exact NULL-free keys per zone maps
			cached =
			    duckdb_mlx::MlxCacheHas(col_keys, total_rows) && duckdb_mlx::MlxGroupbyCachedSafe(group_key, value_key);
		}
	}

	if (has_filter) {
		col_keys.clear();
		cached = false;
	}

	auto mlx_op = make_uniq<MlxGroupbyLogicalOperator>(agg.group_index, agg.aggregate_index, group_col, value_col,
	                                                   std::move(output_types), std::move(col_keys),
	                                                   std::move(table_prefix), total_rows, cached);
	mlx_op->estimated_cardinality = agg.estimated_cardinality;
	mlx_op->has_estimated_cardinality = agg.has_estimated_cardinality;
	if (!cached) {
		mlx_op->children.push_back(std::move(agg.children[0]));
	}
	plan = std::move(mlx_op);
	return true;
}

static bool TryInterceptAggregate(ClientContext &context, unique_ptr<LogicalOperator> &plan, idx_t min_rows) {
	if (plan->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return false;
	}
	auto &agg = plan->Cast<LogicalAggregate>();
	if (!agg.groups.empty() || agg.grouping_sets.size() > 1 ||
	    (!agg.grouping_sets.empty() && !agg.grouping_sets[0].empty())) {
		return false;
	}
	if (agg.expressions.empty()) {
		return false;
	}

	// child shape: [PROJECTION ->] [FILTER ->] GET
	optional_ptr<LogicalProjection> proj;
	optional_ptr<LogicalFilter> filter_op;
	optional_ptr<LogicalGet> get;
	reference<LogicalOperator> node = *agg.children[0];
	if (node.get().type == LogicalOperatorType::LOGICAL_PROJECTION) {
		proj = &node.get().Cast<LogicalProjection>();
		node = *proj->children[0];
	}
	if (node.get().type == LogicalOperatorType::LOGICAL_FILTER) {
		filter_op = &node.get().Cast<LogicalFilter>();
		if (!filter_op->projection_map.empty()) {
			return false;
		}
		node = *filter_op->children[0];
	}
	if (node.get().type != LogicalOperatorType::LOGICAL_GET) {
		return false;
	}
	get = &node.get().Cast<LogicalGet>();
	if (get->function.name != "seq_scan") {
		return false;
	}
	auto estimated_rows =
	    get->has_estimated_cardinality ? get->estimated_cardinality : get->EstimateCardinality(context);
	if (estimated_rows < min_rows) {
		return false;
	}

	// Build GPU cache filter from residual FILTER + table_filters (used only on
	// MLX_SUM_CACHED). Cold path: DuckDB scan applies table_filters; FILTER op
	// stays in the plan for residual predicates.
	MlxFilter cache_filter;
	std::set<int32_t> filter_cols;
	if (filter_op) {
		bool first = true;
		for (auto &expr : filter_op->expressions) {
			MlxExprTranslator translator(*get, nullptr);
			MlxExprTranslator::Lane expr_lane;
			if (!translator.Translate(*expr, expr_lane) || expr_lane != MlxExprTranslator::Lane::BOOL_LANE) {
				return false;
			}
			cache_filter.ops.insert(cache_filter.ops.end(), translator.ops.begin(), translator.ops.end());
			filter_cols.insert(translator.referenced_cols.begin(), translator.referenced_cols.end());
			if (!first) {
				cache_filter.ops.push_back({MlxExprOpCode::AND, 0, 0});
			}
			first = false;
		}
	}
	for (auto &entry : get->table_filters.filters) {
		auto &filter_column_ids = get->GetColumnIds();
		idx_t col_pos = filter_column_ids.size();
		for (idx_t i = 0; i < filter_column_ids.size(); i++) {
			if (filter_column_ids[i].GetPrimaryIndex() == entry.first) {
				col_pos = i;
				break;
			}
		}
		if (col_pos == filter_column_ids.size()) {
			return false;
		}
		vector<MlxExprOp> ops;
		if (!TranslateTableFilter(*entry.second, col_pos, entry.first, *get, ops, filter_cols)) {
			return false;
		}
		if (!ops.empty()) {
			bool had_ops = !cache_filter.ops.empty();
			cache_filter.ops.insert(cache_filter.ops.end(), ops.begin(), ops.end());
			if (had_ops) {
				cache_filter.ops.push_back({MlxExprOpCode::AND, 0, 0});
			}
		}
	}
	cache_filter.null_cols.assign(filter_cols.begin(), filter_cols.end());

	// interval bound of |program| per row from column statistics; declines
	// int-lane sums whose exact int64 accumulation could overflow
	auto program_abs_bound = [&](const MlxSumProgram &program, double &bound) {
		std::vector<double> stack;
		for (auto &op : program.ops) {
			switch (op.code) {
			case MlxExprOpCode::LOAD_COL: {
				unique_ptr<BaseStatistics> stats;
				if (get->function.statistics_extended) {
					auto &column_ids = get->GetColumnIds();
					idx_t pos = column_ids.size();
					for (idx_t i = 0; i < column_ids.size(); i++) {
						if (column_ids[i].GetPrimaryIndex() == static_cast<idx_t>(op.col)) {
							pos = i;
							break;
						}
					}
					if (pos < column_ids.size()) {
						TableFunctionGetStatisticsInput input(get->bind_data.get(), column_ids[pos]);
						stats = get->function.statistics_extended(context, input);
					}
				}
				if (!stats || !NumericStats::HasMinMax(*stats)) {
					return false;
				}
				auto lo = NumericStats::Min(*stats).GetValue<double>();
				auto hi = NumericStats::Max(*stats).GetValue<double>();
				double raw_scale = 1.0;
				auto &col_type = get->returned_types[op.col];
				if (col_type.id() == LogicalTypeId::DECIMAL) {
					raw_scale = static_cast<double>(MlxExprTranslator::Pow10(DecimalType::GetScale(col_type)));
				}
				stack.push_back(std::max(std::abs(lo), std::abs(hi)) * raw_scale);
				break;
			}
			case MlxExprOpCode::CONST_VAL:
				stack.push_back(op.int_lane ? std::abs(static_cast<double>(op.ivalue)) : std::abs(op.value));
				break;
			case MlxExprOpCode::ADD:
			case MlxExprOpCode::SUB: {
				auto b = stack.back();
				stack.pop_back();
				stack.back() += b;
				break;
			}
			case MlxExprOpCode::MUL: {
				auto b = stack.back();
				stack.pop_back();
				stack.back() *= b;
				break;
			}
			case MlxExprOpCode::NEGATE:
			case MlxExprOpCode::ABS:
				break;
			default:
				return false;
			}
		}
		bound = stack.empty() ? 0.0 : stack.back();
		return true;
	};

	vector<MlxSumProgram> programs;
	vector<LogicalType> agg_types;
	for (auto &expr : agg.expressions) {
		if (expr->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			return false;
		}
		auto &aggr = expr->Cast<BoundAggregateExpression>();
		if (aggr.IsDistinct() || aggr.filter) {
			return false;
		}
		MlxSumProgram program;
		auto &name = aggr.function.name;
		if (name == "count_star" && aggr.children.empty()) {
			program.kind = MlxAggKind::COUNT_STAR;
		} else if (name == "count" && aggr.children.size() == 1) {
			program.kind = MlxAggKind::COUNT;
		} else if (aggr.children.size() == 1) {
			if (name == "sum" || name == "sum_no_overflow") {
				program.kind = MlxAggKind::SUM;
			} else if (name == "avg") {
				program.kind = MlxAggKind::AVG;
			} else if (name == "min") {
				program.kind = MlxAggKind::MIN;
			} else if (name == "max") {
				program.kind = MlxAggKind::MAX;
			} else {
				return false;
			}
		} else {
			return false;
		}
		if (!aggr.children.empty()) {
			MlxExprTranslator translator(*get, proj);
			MlxExprTranslator::Lane lane;
			if (!translator.Translate(*aggr.children[0], lane) || lane == MlxExprTranslator::Lane::BOOL_LANE) {
				return false;
			}
			program.int_lane = lane == MlxExprTranslator::Lane::INT_LANE;
			program.ops = std::move(translator.ops);
			program.null_cols.assign(translator.referenced_cols.begin(), translator.referenced_cols.end());
		}

		// result typing: exact DECIMAL results ride the int lane
		if (program.kind == MlxAggKind::COUNT || program.kind == MlxAggKind::COUNT_STAR) {
			if (aggr.return_type.id() != LogicalTypeId::BIGINT) {
				return false;
			}
			agg_types.push_back(LogicalType::BIGINT);
		} else if (program.int_lane) {
			auto &child_type = aggr.children[0]->return_type;
			if (child_type.id() != LogicalTypeId::DECIMAL) {
				return false;
			}
			auto child_scale = DecimalType::GetScale(child_type);
			switch (program.kind) {
			case MlxAggKind::SUM:
				if (aggr.return_type.id() != LogicalTypeId::DECIMAL ||
				    DecimalType::GetScale(aggr.return_type) != child_scale) {
					return false;
				}
				agg_types.push_back(aggr.return_type);
				break;
			case MlxAggKind::AVG:
				if (aggr.return_type.id() != LogicalTypeId::DOUBLE) {
					return false;
				}
				program.render_scale = 1.0 / static_cast<double>(MlxExprTranslator::Pow10(child_scale));
				agg_types.push_back(LogicalType::DOUBLE);
				break;
			default: // MIN / MAX keep the child's decimal type
				if (aggr.return_type != child_type) {
					return false;
				}
				agg_types.push_back(aggr.return_type);
				break;
			}
			if (program.kind == MlxAggKind::SUM || program.kind == MlxAggKind::AVG) {
				double bound = 0;
				if (!program_abs_bound(program, bound) ||
				    bound * static_cast<double>(std::max<idx_t>(estimated_rows, 1)) >= std::ldexp(1.0, 61)) {
					return false; // exact int64 accumulation could overflow
				}
			}
		} else {
			if (aggr.return_type.id() != LogicalTypeId::DOUBLE) {
				return false;
			}
			agg_types.push_back(LogicalType::DOUBLE);
		}
		programs.push_back(std::move(program));
	}

	// DuckDB 1.5.4 column model: scan projection carries agg columns only;
	// cache projection also includes filter columns for MLX_SUM_CACHED masks.
	auto &column_ids = get->GetColumnIds();
	auto storage_to_column_ids_idx = [&](idx_t storage_col) -> idx_t {
		for (idx_t i = 0; i < column_ids.size(); i++) {
			if (column_ids[i].GetPrimaryIndex() == storage_col) {
				return i;
			}
		}
		return column_ids.size();
	};

	std::set<idx_t> agg_column_ids;
	CollectProgramColumnIds(programs, *get, agg_column_ids);
	if (agg_column_ids.empty() && programs.size() == 1 && programs[0].kind == MlxAggKind::COUNT_STAR) {
		// count_star has no column refs; scan at least one column for row count
		if (!column_ids.empty()) {
			agg_column_ids.insert(0);
		}
	}

	vector<idx_t> scan_projection;
	if (get->projection_ids.empty()) {
		scan_projection.assign(agg_column_ids.begin(), agg_column_ids.end());
	} else {
		std::set<idx_t> projection_set(get->projection_ids.begin(), get->projection_ids.end());
		for (auto column_ids_idx : agg_column_ids) {
			projection_set.insert(column_ids_idx);
		}
		scan_projection.assign(projection_set.begin(), projection_set.end());
	}

	std::set<idx_t> cache_column_ids(agg_column_ids.begin(), agg_column_ids.end());
	for (auto storage_col : filter_cols) {
		auto column_ids_idx = storage_to_column_ids_idx(static_cast<idx_t>(storage_col));
		if (column_ids_idx >= column_ids.size()) {
			return false;
		}
		cache_column_ids.insert(column_ids_idx);
	}
	vector<idx_t> cache_projection(scan_projection.begin(), scan_projection.end());
	for (auto column_ids_idx : cache_column_ids) {
		if (std::find(cache_projection.begin(), cache_projection.end(), column_ids_idx) == cache_projection.end()) {
			cache_projection.push_back(column_ids_idx);
		}
	}

	if (!RemapStorageToChunk(cache_filter.ops, cache_filter.null_cols, cache_projection, *get)) {
		return false;
	}
	for (auto &program : programs) {
		if (!RemapStorageToChunk(program.ops, program.null_cols, scan_projection, *get)) {
			return false;
		}
	}

	bool has_table_filters = !get->table_filters.filters.empty();

	// GPU column cache identity: catalog.schema.table plus each output
	// column's storage id. A row-count match against current storage decides
	// whether the cached population is served (GQE-style resident table).
	vector<string> col_keys;
	string table_prefix;
	int64_t total_rows = 0;
	bool cached = false;
	auto table = get->GetTable();
	if (table) {
		table_prefix = table->ParentCatalog().GetName() + "." + table->ParentSchema().name + "." + table->name + "#";
		total_rows = NumericCast<int64_t>(table->GetStorage().GetTotalRows());
		if (cache_projection.empty()) {
			for (auto &col : column_ids) {
				col_keys.push_back(table_prefix + std::to_string(col.GetPrimaryIndex()));
			}
		} else {
			for (auto pid : cache_projection) {
				col_keys.push_back(table_prefix + std::to_string(column_ids[pid].GetPrimaryIndex()));
			}
		}
		cached = duckdb_mlx::MlxCacheHas(col_keys, total_rows);
	}

	// Plain single-column aggs are scan-bound on CPU — always decline (cold or cached).
	if (ProgramsAreCpuFast(programs)) {
		return false;
	}

	// Sirius-style: leave table_filters on GET; extend scan projection for agg
	// columns only; keep FILTER in the child chain on the cold path.
	get->projection_ids = std::move(scan_projection);
	bool skip_cache_populate = has_table_filters;

	auto mlx_op = make_uniq<MlxSumLogicalOperator>(agg.aggregate_index, std::move(programs), std::move(agg_types),
	                                               std::move(cache_filter), std::move(col_keys),
	                                               std::move(table_prefix), total_rows, cached, skip_cache_populate);
	if (!cached) {
		mlx_op->children.push_back(std::move(agg.children[0]));
	}
	plan = std::move(mlx_op);
	duckdb_mlx::LogDebug(cached ? "MLX_SUM serving from the GPU-resident cache (no scan)"
	                            : "MLX_SUM intercepted an ungrouped aggregation");
	return true;
}

static void WalkPlan(ClientContext &context, unique_ptr<LogicalOperator> &plan, idx_t min_rows) {
	for (auto &child : plan->children) {
		WalkPlan(context, child, min_rows);
	}
	if (!TryInterceptGroupBy(context, plan, min_rows)) {
		TryInterceptAggregate(context, plan, min_rows);
	}
}

static void MlxOptimizeFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	Value enabled;
	if (!input.context.TryGetCurrentSetting("mlx_execution", enabled) || !enabled.GetValue<bool>()) {
		return;
	}
	Value min_rows_value;
	idx_t min_rows = 524288;
	if (input.context.TryGetCurrentSetting("mlx_min_rows", min_rows_value)) {
		min_rows = min_rows_value.GetValue<idx_t>();
	}
	WalkPlan(input.context, plan, min_rows);
}

void RegisterMlxOptimizer(DatabaseInstance &db) {
	OptimizerExtension ext;
	ext.optimize_function = MlxOptimizeFunction;
	OptimizerExtension::Register(DBConfig::GetConfig(db), ext);
}

} // namespace duckdb
