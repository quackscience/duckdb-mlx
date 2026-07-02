#include "mlx_transparent.hpp"

#include "duckdb.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/execution/physical_operator.hpp"
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
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry_retriever.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
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
	MlxExprTranslator(LogicalGet &get, vector<LogicalProjection *> projs_p) : get(get), projs(std::move(projs_p)) {
	}
	MlxExprTranslator(LogicalGet &get, optional_ptr<LogicalProjection> proj) : get(get) {
		if (proj) {
			projs.push_back(proj.get());
		}
	}

	LogicalGet &get;
	vector<LogicalProjection *> projs; // top-to-bottom chain below the aggregate
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
			for (auto proj : projs) {
				if (proj && colref.binding.table_index == proj->table_index) {
					return Translate(*proj->expressions[colref.binding.column_index], lane);
				}
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
			duckdb_mlx::MlxCacheFuseTable(table_prefix);
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
//===--------------------------------------------------------------------===//
// Generalized grouped operator (TPC-H Q1 class): 1-2 dense keys (int ranges
// or dictionary-encoded VARCHAR), any supported aggregate programs, exact
// lanes. GPU evaluates codes/masks/expressions; host threads merge exactly.
//===--------------------------------------------------------------------===//

struct MlxGroupKeySpec {
	int32_t scan_col = 0;   // chunk position after remap
	LogicalType type;       // output type (compressed when materialization fired)
	int64_t offset = 0;     // int keys: stats min (dense-code base)
	int64_t emit_base = 0;  // added to the dense code when emitting the key
	int64_t card = 0;       // plan-time estimate; refined at execution
	bool is_varchar = false;
	string cache_key;       // cache/dictionary identity
};

class MlxGroupedGlobalSinkState : public GlobalSinkState {
public:
	mutex glock;
	vector<MlxSumSegment> segments;
	// local dictionary for VARCHAR keys (published at cache-populate time)
	mutex dict_lock;
	vector<vector<string>> dict_strings;                    // per key
	vector<std::unordered_map<string, int32_t>> dict_index; // per key
	duckdb_mlx::MlxGroupedState state;
	bool state_ready = false;
};

class MlxGroupedLocalSinkState : public LocalSinkState {
public:
	MlxSumSegment segment;
};

class MlxGroupedGlobalSourceState : public GlobalSourceState {
public:
	idx_t offset = 0;
	duckdb_mlx::MlxGroupedState state;
	vector<vector<string>> dicts; // decode tables per key
	vector<int64_t> cards;
	bool ready = false;
};

class MlxGroupedPhysicalOperator : public PhysicalOperator {
public:
	MlxGroupedPhysicalOperator(PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality,
	                           vector<MlxGroupKeySpec> keys_p, vector<MlxSumProgram> programs_p,
	                           MlxFilter cache_filter_p, vector<string> col_keys_p, string table_prefix_p,
	                           int64_t expected_rows_p, bool cached_p, bool skip_cache_populate_p,
	                           vector<int32_t> output_map_p)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
	      keys(std::move(keys_p)), programs(std::move(programs_p)), cache_filter(std::move(cache_filter_p)),
	      col_keys(std::move(col_keys_p)), table_prefix(std::move(table_prefix_p)), expected_rows(expected_rows_p),
	      cached(cached_p), skip_cache_populate(skip_cache_populate_p), output_map(std::move(output_map_p)) {
	}

	vector<MlxGroupKeySpec> keys;
	vector<MlxSumProgram> programs;
	MlxFilter cache_filter;
	vector<string> col_keys;
	string table_prefix;
	int64_t expected_rows = 0;
	bool cached;
	bool skip_cache_populate;
	//! output order: entry < nkeys => key, else program (nkeys + p). Mode B
	//! (absorbed decompress projection) reorders outputs arbitrarily.
	vector<int32_t> output_map;

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
		return make_uniq<MlxGroupedLocalSinkState>();
	}
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		auto state = make_uniq<MlxGroupedGlobalSinkState>();
		state->dict_strings.resize(keys.size());
		state->dict_index.resize(keys.size());
		return std::move(state);
	}
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override {
		return make_uniq<MlxGroupedGlobalSourceState>();
	}

	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override {
		auto &segment = input.local_state.Cast<MlxGroupedLocalSinkState>().segment;
		auto &gstate = input.global_state.Cast<MlxGroupedGlobalSinkState>();
		if (segment.values.empty()) {
			segment.values.resize(chunk.ColumnCount());
			segment.ivalues.resize(chunk.ColumnCount());
			segment.valid.resize(chunk.ColumnCount());
		}
		auto count = chunk.size();
		for (idx_t col = 0; col < chunk.ColumnCount(); col++) {
			// VARCHAR keys are dictionary-encoded into fp32 codes
			int64_t key_idx = -1;
			for (size_t k = 0; k < keys.size(); k++) {
				if (keys[k].is_varchar && keys[k].scan_col == NumericCast<int32_t>(col)) {
					key_idx = NumericCast<int64_t>(k);
					break;
				}
			}
			UnifiedVectorFormat fmt;
			chunk.data[col].ToUnifiedFormat(count, fmt);
			if (key_idx >= 0) {
				auto strings = UnifiedVectorFormat::GetData<string_t>(fmt);
				auto &values = segment.values[col];
				auto base = values.size();
				values.resize(base + count);
				lock_guard<mutex> guard(gstate.dict_lock);
				auto &dict = gstate.dict_index[key_idx];
				auto &decode = gstate.dict_strings[key_idx];
				for (idx_t i = 0; i < count; i++) {
					auto idx = fmt.sel->get_index(i);
					if (!fmt.validity.RowIsValid(idx)) {
						values[base + i] = -1.0f; // dumped by the code guard
						continue;
					}
					auto str = strings[idx].GetString();
					auto it = dict.find(str);
					int32_t code;
					if (it != dict.end()) {
						code = it->second;
					} else {
						code = NumericCast<int32_t>(decode.size());
						decode.push_back(str);
						dict.emplace(str, code);
					}
					values[base + i] = static_cast<float>(code);
				}
			} else if (!AppendLaneColumn(chunk.data[col].GetType(), fmt, count, segment.values[col],
			                             segment.ivalues[col], segment.valid[col])) {
				throw InternalException("MLX_GROUPBY: unexpected column type at execution time");
			}
		}
		return SinkResultType::NEED_MORE_INPUT;
	}

	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override {
		auto &lstate = input.local_state.Cast<MlxGroupedLocalSinkState>();
		auto &gstate = input.global_state.Cast<MlxGroupedGlobalSinkState>();
		if (lstate.segment.values.empty()) {
			return SinkCombineResultType::FINISHED;
		}
		lock_guard<mutex> guard(gstate.glock);
		gstate.segments.push_back(std::move(lstate.segment));
		return SinkCombineResultType::FINISHED;
	}

	duckdb_mlx::MlxGroupedSpec BuildSpec(const vector<int64_t> &cards) const {
		duckdb_mlx::MlxGroupedSpec spec;
		for (size_t k = 0; k < keys.size(); k++) {
			spec.key_cols.push_back(keys[k].scan_col);
			spec.key_offsets.push_back(keys[k].is_varchar ? 0 : keys[k].offset);
			spec.key_cards.push_back(cards[k]);
		}
		return spec;
	}

	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override {
		auto &gstate = input.global_state.Cast<MlxGroupedGlobalSinkState>();
		// runtime-exact cards: dictionaries as built, int ranges from stats
		vector<int64_t> cards;
		for (size_t k = 0; k < keys.size(); k++) {
			cards.push_back(keys[k].is_varchar
			                    ? std::max<int64_t>(1, NumericCast<int64_t>(gstate.dict_strings[k].size()))
			                    : keys[k].card);
		}
		auto spec = BuildSpec(cards);
		duckdb_mlx::MlxFilter no_filter; // the cold scan already applied filters
		gstate.state = duckdb_mlx::MlxGroupedInit(spec, programs);
		for (auto &segment : gstate.segments) {
			vector<MlxColumnData> cols;
			size_t row_count = SegmentRowCount(segment);
			SegmentColumns(segment, cols);
			duckdb_mlx::MlxGroupedAccumulate(gstate.state, spec, cols, row_count, programs, no_filter);
		}
		duckdb_mlx::MlxGroupedGpuFinish(gstate.state, programs);
		gstate.state_ready = true;

		if (!col_keys.empty() && !skip_cache_populate) {
			auto plan = duckdb_mlx::MlxCacheBeginPopulation(table_prefix, col_keys, expected_rows);
			for (auto &segment : gstate.segments) {
				vector<MlxColumnData> cols;
				size_t row_count = SegmentRowCount(segment);
				SegmentColumns(segment, cols);
				duckdb_mlx::MlxCacheStoreSegment(plan.population, col_keys, plan.store_col, cols, row_count);
			}
			duckdb_mlx::MlxCacheFuseTable(table_prefix);
			for (size_t k = 0; k < keys.size(); k++) {
				if (keys[k].is_varchar) {
					duckdb_mlx::MlxDictInstall(keys[k].cache_key, plan.population, gstate.dict_strings[k]);
				}
			}
			duckdb_mlx::LogDebug("MLX_GROUPBY populated the GPU cache for " + table_prefix);
		}
		return SinkFinalizeType::READY;
	}

	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override {
		auto &source = input.global_state.Cast<MlxGroupedGlobalSourceState>();
		if (!source.ready) {
			source.dicts.resize(keys.size());
			if (cached) {
				for (size_t k = 0; k < keys.size(); k++) {
					if (keys[k].is_varchar) {
						auto population = duckdb_mlx::MlxCachePopulation(keys[k].cache_key);
						source.dicts[k] = duckdb_mlx::MlxDictStrings(keys[k].cache_key, population);
						source.cards.push_back(std::max<int64_t>(1, NumericCast<int64_t>(source.dicts[k].size())));
					} else {
						source.cards.push_back(keys[k].card);
					}
				}
				auto spec = BuildSpec(source.cards);
				source.state = duckdb_mlx::MlxGroupedInit(spec, programs);
				duckdb_mlx::MlxGroupedAccumulateCached(source.state, spec, col_keys, programs, cache_filter,
				                                       table_prefix);
				duckdb_mlx::MlxGroupedGpuFinish(source.state, programs);
			} else {
				auto &gstate = sink_state->Cast<MlxGroupedGlobalSinkState>();
				source.state = gstate.state;
				for (size_t k = 0; k < keys.size(); k++) {
					source.dicts[k] = gstate.dict_strings[k];
					source.cards.push_back(keys[k].is_varchar
					                           ? std::max<int64_t>(1, NumericCast<int64_t>(source.dicts[k].size()))
					                           : keys[k].card);
				}
			}
			source.ready = true;
		}

		auto &state = source.state;
		auto nkeys = keys.size();
		auto nprogs = programs.size();
		idx_t out = 0;
		while (source.offset < static_cast<idx_t>(state.card) && out < STANDARD_VECTOR_SIZE) {
			auto g = static_cast<int64_t>(source.offset++);
			if (state.rows[g] == 0) {
				continue;
			}
			// decompose the combined code into per-key indexes
			vector<int64_t> key_codes(nkeys);
			auto rest = g;
			for (size_t k = nkeys; k > 0; k--) {
				key_codes[k - 1] = rest % source.cards[k - 1];
				rest /= source.cards[k - 1];
			}
			for (idx_t o = 0; o < output_map.size(); o++) {
				auto &vec = chunk.data[o];
				auto entry = output_map[o];
				if (entry < NumericCast<int32_t>(nkeys)) {
					auto k = static_cast<size_t>(entry);
					if (keys[k].is_varchar) {
						FlatVector::GetData<string_t>(vec)[out] =
						    StringVector::AddString(vec, source.dicts[k][key_codes[k]]);
					} else {
						auto raw = key_codes[k] + keys[k].emit_base;
						switch (vec.GetType().InternalType()) {
						case PhysicalType::INT64:
							FlatVector::GetData<int64_t>(vec)[out] = raw;
							break;
						case PhysicalType::INT32:
							FlatVector::GetData<int32_t>(vec)[out] = NumericCast<int32_t>(raw);
							break;
						case PhysicalType::UINT32:
							FlatVector::GetData<uint32_t>(vec)[out] = NumericCast<uint32_t>(raw);
							break;
						case PhysicalType::UINT16:
							FlatVector::GetData<uint16_t>(vec)[out] = NumericCast<uint16_t>(raw);
							break;
						case PhysicalType::UINT8:
							FlatVector::GetData<uint8_t>(vec)[out] = NumericCast<uint8_t>(raw);
							break;
						default:
							throw InternalException("MLX_GROUPBY: unexpected key type");
						}
					}
					continue;
				}
				auto p = static_cast<size_t>(entry) - nkeys;
				auto slot = static_cast<size_t>(g) * nprogs + p;
				auto count = state.counts[slot];
				switch (programs[p].kind) {
				case MlxAggKind::COUNT_STAR:
					FlatVector::GetData<int64_t>(vec)[out] = state.rows[g];
					break;
				case MlxAggKind::COUNT:
					FlatVector::GetData<int64_t>(vec)[out] = count;
					break;
				default:
					if (count == 0) {
						FlatVector::Validity(vec).SetInvalid(out);
						break;
					}
					if (programs[p].int_lane && vec.GetType().id() == LogicalTypeId::DECIMAL) {
						auto iv = state.ivalues[slot];
						switch (vec.GetType().InternalType()) {
						case PhysicalType::INT128: {
							hugeint_t h;
							h.lower = static_cast<uint64_t>(iv);
							h.upper = static_cast<int64_t>(iv >> 64);
							FlatVector::GetData<hugeint_t>(vec)[out] = h;
							break;
						}
						case PhysicalType::INT64:
							FlatVector::GetData<int64_t>(vec)[out] = static_cast<int64_t>(iv);
							break;
						case PhysicalType::INT32:
							FlatVector::GetData<int32_t>(vec)[out] = static_cast<int32_t>(iv);
							break;
						default:
							FlatVector::GetData<int16_t>(vec)[out] = static_cast<int16_t>(iv);
							break;
						}
					} else {
						double value = programs[p].int_lane ? static_cast<double>(state.ivalues[slot])
						                                    : state.fvalues[slot];
						if (programs[p].kind == MlxAggKind::AVG) {
							value /= static_cast<double>(count);
						}
						FlatVector::GetData<double>(vec)[out] = value * programs[p].render_scale;
					}
					break;
				}
			}
			out++;
		}
		chunk.SetCardinality(out);
		return source.offset >= static_cast<idx_t>(state.card) && out == 0 ? SourceResultType::FINISHED
		                                                                   : SourceResultType::HAVE_MORE_OUTPUT;
	}
};

class MlxGroupedLogicalOperator : public LogicalExtensionOperator {
public:
	MlxGroupedLogicalOperator(idx_t group_index_p, idx_t aggregate_index_p, vector<MlxGroupKeySpec> keys_p,
	                          vector<MlxSumProgram> programs_p, vector<LogicalType> agg_types_p,
	                          MlxFilter cache_filter_p, vector<string> col_keys_p, string table_prefix_p,
	                          int64_t expected_rows_p, bool cached_p, bool skip_cache_populate_p)
	    : group_index(group_index_p), aggregate_index(aggregate_index_p), keys(std::move(keys_p)),
	      programs(std::move(programs_p)), agg_types(std::move(agg_types_p)),
	      cache_filter(std::move(cache_filter_p)), col_keys(std::move(col_keys_p)),
	      table_prefix(std::move(table_prefix_p)), expected_rows(expected_rows_p), cached(cached_p),
	      skip_cache_populate(skip_cache_populate_p) {
		// natural order: keys then programs; mode B installs its own map
		for (idx_t i = 0; i < keys.size() + programs.size(); i++) {
			output_map.push_back(NumericCast<int32_t>(i));
		}
	}

	idx_t group_index;
	idx_t aggregate_index;
	//! mode B: outputs bind to the absorbed parent projection's table index
	//! in output_map order; otherwise two binding spaces (groups, aggregates)
	bool single_binding = false;
	idx_t output_index = 0;
	vector<int32_t> output_map;
	vector<LogicalType> output_types;
	vector<MlxGroupKeySpec> keys;
	vector<MlxSumProgram> programs;
	vector<LogicalType> agg_types;
	MlxFilter cache_filter;
	vector<string> col_keys;
	string table_prefix;
	int64_t expected_rows;
	bool cached;
	bool skip_cache_populate;

	string GetName() const override {
		return cached ? "MLX_GROUPBY_CACHED" : "MLX_GROUPBY";
	}
	string GetExtensionName() const override {
		return "duckdb_mlx";
	}

	vector<ColumnBinding> GetColumnBindings() override {
		vector<ColumnBinding> bindings;
		if (single_binding) {
			for (idx_t i = 0; i < output_map.size(); i++) {
				bindings.emplace_back(output_index, i);
			}
			return bindings;
		}
		for (idx_t i = 0; i < keys.size(); i++) {
			bindings.emplace_back(group_index, i);
		}
		for (idx_t i = 0; i < programs.size(); i++) {
			bindings.emplace_back(aggregate_index, i);
		}
		return bindings;
	}

	void ResolveTypes() override {
		types.clear();
		for (auto entry : output_map) {
			if (entry < NumericCast<int32_t>(keys.size())) {
				types.push_back(keys[entry].type);
			} else {
				types.push_back(agg_types[entry - NumericCast<int32_t>(keys.size())]);
			}
		}
	}

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override {
		auto &op = planner.Make<MlxGroupedPhysicalOperator>(types, estimated_cardinality, std::move(keys),
		                                                    std::move(programs), std::move(cache_filter),
		                                                    std::move(col_keys), std::move(table_prefix),
		                                                    expected_rows, cached, skip_cache_populate,
		                                                    std::move(output_map));
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

//! Cost gate for the exact int64 lane: 64-bit arithmetic is emulated on
//! Apple GPUs and the CPU's filtered scan reads only surviving row groups,
//! so a lone decimal aggregate (TPC-H Q6 class) loses to the CPU. Multi-
//! aggregate decimal queries (Q1 class) amortize the column reads and win.
static bool IntLaneCpuFast(const vector<duckdb_mlx::MlxSumProgram> &programs) {
	idx_t int_lane_values = 0;
	idx_t value_programs = 0;
	for (auto &program : programs) {
		if (program.kind == duckdb_mlx::MlxAggKind::COUNT_STAR) {
			continue;
		}
		value_programs++;
		if (program.int_lane) {
			int_lane_values++;
		}
	}
	return int_lane_values > 0 && value_programs < 3;
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

//! Shared aggregate-program acceptance for the ungrouped and grouped
//! interceptors: translation, lanes, result typing and the int64
//! overflow gate.
static bool BuildAggregatePrograms(ClientContext &context, optional_ptr<LogicalGet> get,
                                   const vector<LogicalProjection *> &projs, LogicalAggregate &agg,
                                   idx_t estimated_rows, bool host_accumulated, vector<MlxSumProgram> &programs,
                                   vector<LogicalType> &agg_types) {
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
					{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
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
				{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
			}
		}
		bound = stack.empty() ? 0.0 : stack.back();
		return true;
	};

	programs.clear();
	agg_types.clear();
	for (auto &expr : agg.expressions) {
		if (expr->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
		}
		auto &aggr = expr->Cast<BoundAggregateExpression>();
		if (aggr.IsDistinct() || aggr.filter) {
			{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
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
				{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
			}
		} else {
			{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
		}
		if (!aggr.children.empty()) {
			MlxExprTranslator translator(*get, projs);
			MlxExprTranslator::Lane lane;
			if (!translator.Translate(*aggr.children[0], lane) || lane == MlxExprTranslator::Lane::BOOL_LANE) {
				{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
			}
			program.int_lane = lane == MlxExprTranslator::Lane::INT_LANE;
			program.ops = std::move(translator.ops);
			program.null_cols.assign(translator.referenced_cols.begin(), translator.referenced_cols.end());
		}

		// result typing: exact DECIMAL results ride the int lane
		if (program.kind == MlxAggKind::COUNT || program.kind == MlxAggKind::COUNT_STAR) {
			if (aggr.return_type.id() != LogicalTypeId::BIGINT) {
				{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
			}
			agg_types.push_back(LogicalType::BIGINT);
		} else if (program.int_lane) {
			auto &child_type = aggr.children[0]->return_type;
			if (child_type.id() != LogicalTypeId::DECIMAL) {
				{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
			}
			auto child_scale = DecimalType::GetScale(child_type);
			switch (program.kind) {
			case MlxAggKind::SUM:
				if (aggr.return_type.id() != LogicalTypeId::DECIMAL ||
				    DecimalType::GetScale(aggr.return_type) != child_scale) {
					{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
				}
				agg_types.push_back(aggr.return_type);
				break;
			case MlxAggKind::AVG:
				if (aggr.return_type.id() != LogicalTypeId::DOUBLE) {
					{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
				}
				program.render_scale = 1.0 / static_cast<double>(MlxExprTranslator::Pow10(child_scale));
				agg_types.push_back(LogicalType::DOUBLE);
				break;
			default: // MIN / MAX keep the child's decimal type
				if (aggr.return_type != child_type) {
					{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
				}
				agg_types.push_back(aggr.return_type);
				break;
			}
			if (program.kind == MlxAggKind::SUM || program.kind == MlxAggKind::AVG) {
				double bound = 0;
				if (!program_abs_bound(program, bound) ||
				    bound * static_cast<double>(std::max<idx_t>(estimated_rows, 1)) >= std::ldexp(1.0, 61)) {
					{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; } // exact int64 accumulation could overflow
				}
				// Metal scatter supports int32/float32 only — route int-lane SUM/AVG to
				// GPU with chunked partial merge into int128 host accumulators.
				program.row_abs_bound = bound;
				constexpr double kInt32Max = static_cast<double>(std::numeric_limits<int32_t>::max());
				if (bound <= kInt32Max) {
					program.int32_rows = true;
				} else {
					program.gpu_fp32_rows = true;
				}
			}
		} else {
			if (aggr.return_type.id() != LogicalTypeId::DOUBLE) {
				{ duckdb_mlx::LogDebug("MLX_AGGPROG decline @" + std::to_string(__LINE__)); return false; }
			}
			agg_types.push_back(LogicalType::DOUBLE);
		}
		programs.push_back(std::move(program));
	}
	return true;
}

static bool TryInterceptGroupBy(ClientContext &context, unique_ptr<LogicalOperator> &plan, idx_t min_rows) {
	// mode B: a decompress projection directly above the aggregate is
	// absorbed, so string-compressed group keys can be emitted as raw
	// dictionary strings
	optional_ptr<LogicalProjection> parent_proj;
	LogicalOperator *agg_node = plan.get();
	if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION && !plan->children.empty() &&
	    plan->children[0]->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		parent_proj = &plan->Cast<LogicalProjection>();
		agg_node = plan->children[0].get();
	}
	if (agg_node->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return false;
	}
	auto &agg = agg_node->Cast<LogicalAggregate>();
	if (agg.groups.empty() || agg.groups.size() > 2 || agg.expressions.empty()) {
		return false;
	}
	if (agg.grouping_sets.size() > 1 ||
	    (!agg.grouping_sets.empty() && agg.grouping_sets[0].size() != agg.groups.size())) {
		return false;
	}

	// walk the projection chain below the aggregate
	vector<LogicalProjection *> projs;
	optional_ptr<LogicalFilter> filter_op;
	optional_ptr<LogicalGet> get;
	reference<LogicalOperator> node = *agg.children[0];
	while (node.get().type == LogicalOperatorType::LOGICAL_PROJECTION && projs.size() < 3) {
		projs.push_back(&node.get().Cast<LogicalProjection>());
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
	auto &column_ids = get->GetColumnIds();

	auto column_stats = [&](idx_t storage_col) -> unique_ptr<BaseStatistics> {
		if (!get->function.statistics_extended) {
			return nullptr;
		}
		for (idx_t i = 0; i < column_ids.size(); i++) {
			if (column_ids[i].GetPrimaryIndex() == storage_col) {
				TableFunctionGetStatisticsInput input(get->bind_data.get(), column_ids[i]);
				return get->function.statistics_extended(context, input);
			}
		}
		return nullptr;
	};

	// group keys: resolve each through the projection chain, unwrapping
	// CompressedMaterialization wrappers (integral: emit compressed values;
	// string: dictionary-encode the raw column and absorb the decompress
	// parent). Keys must be provably NULL-free.
	vector<MlxGroupKeySpec> keys;
	std::set<int32_t> key_storage_cols;
	int64_t combined_card = 1;
	bool any_string_compressed = false;
	for (auto &group : agg.groups) {
		const Expression *cur = group.get();
		int64_t compress_offset = 0;
		bool string_compressed = false;
		LogicalType output_type;
		bool output_type_set = false;
		while (true) {
			if (cur->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
				auto &cref = cur->Cast<BoundColumnRefExpression>();
				LogicalProjection *owner = nullptr;
				for (auto proj : projs) {
					if (cref.binding.table_index == proj->table_index) {
						owner = proj;
						break;
					}
				}
				if (!owner) {
					break; // base column (or unknown -> checked below)
				}
				cur = owner->expressions[cref.binding.column_index].get();
				continue;
			}
			if (cur->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
				auto &fn = cur->Cast<BoundFunctionExpression>();
				if (StringUtil::StartsWith(fn.function.name, "__internal_compress_integral_") &&
				    fn.children.size() == 2 &&
				    fn.children[1]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
					compress_offset =
					    fn.children[1]->Cast<BoundConstantExpression>().value.GetValue<int64_t>();
					if (!output_type_set) {
						output_type = fn.return_type;
						output_type_set = true;
					}
					cur = fn.children[0].get();
					continue;
				}
				if (StringUtil::StartsWith(fn.function.name, "__internal_compress_string_") &&
				    fn.children.size() == 1) {
					string_compressed = true;
					cur = fn.children[0].get();
					continue;
				}
				return false;
			}
			return false;
		}
		if (cur->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			return false;
		}
		auto &cref = cur->Cast<BoundColumnRefExpression>();
		if (cref.binding.table_index != get->table_index || cref.binding.column_index >= column_ids.size()) {
			return false;
		}
		if (string_compressed) {
			if (!parent_proj) {
				return false; // the decompress parent must be absorbed
			}
			any_string_compressed = true;
		}
		auto storage_col = column_ids[cref.binding.column_index].GetPrimaryIndex();
		auto stats = column_stats(storage_col);
		if (!stats || stats->CanHaveNull()) {
			return false;
		}
		MlxGroupKeySpec key;
		key.scan_col = NumericCast<int32_t>(storage_col); // remapped below
		key.type = output_type_set ? output_type : cref.return_type;
		switch (cref.return_type.id()) {
		case LogicalTypeId::BIGINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::UTINYINT:
		case LogicalTypeId::USMALLINT:
		case LogicalTypeId::UINTEGER: {
			if (!NumericStats::HasMinMax(*stats)) {
				return false;
			}
			auto lo = NumericStats::Min(*stats).GetValue<int64_t>();
			auto hi = NumericStats::Max(*stats).GetValue<int64_t>();
			if (hi < lo || hi - lo + 1 > 65536) {
				return false;
			}
			key.offset = lo;
			key.emit_base = lo - compress_offset;
			key.card = hi - lo + 1;
			break;
		}
		case LogicalTypeId::VARCHAR: {
			auto distinct = stats->GetDistinctCount();
			if (distinct == 0 || distinct > 4096) {
				return false;
			}
			key.is_varchar = true;
			key.type = LogicalType::VARCHAR; // raw strings (decompress absorbed)
			key.card = NumericCast<int64_t>(distinct);
			break;
		}
		default:
			return false;
		}
		combined_card *= key.card;
		if (combined_card > 65536) {
			return false;
		}
		key_storage_cols.insert(NumericCast<int32_t>(storage_col));
		keys.push_back(std::move(key));
	}
	// mode B is only engaged for string-compressed keys; otherwise intercept
	// at the aggregate node itself (the parent projection stays)
	if (parent_proj && !any_string_compressed) {
		return false;
	}

	// mode B: map the parent projection's outputs onto our keys/programs
	vector<int32_t> output_map;
	if (parent_proj) {
		for (auto &pexpr : parent_proj->expressions) {
			const Expression *e = pexpr.get();
			if (e->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
				auto &fn = e->Cast<BoundFunctionExpression>();
				if (!StringUtil::StartsWith(fn.function.name, "__internal_decompress_string") ||
				    fn.children.empty()) {
					return false;
				}
				e = fn.children[0].get();
			}
			if (e->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				return false;
			}
			auto &cref = e->Cast<BoundColumnRefExpression>();
			if (cref.binding.table_index == agg.group_index) {
				if (cref.binding.column_index >= keys.size()) {
					return false;
				}
				output_map.push_back(NumericCast<int32_t>(cref.binding.column_index));
			} else if (cref.binding.table_index == agg.aggregate_index) {
				if (cref.binding.column_index >= agg.expressions.size()) {
					return false;
				}
				output_map.push_back(NumericCast<int32_t>(keys.size() + cref.binding.column_index));
			} else {
				return false;
			}
		}
	}

	// WHERE predicate (Sirius-style: cold scan applies it; GPU mask on cached)
	MlxFilter cache_filter;
	std::set<int32_t> filter_cols;
	if (filter_op) {
		bool first = true;
		for (auto &expr : filter_op->expressions) {
			MlxExprTranslator translator(*get, vector<LogicalProjection *> {});
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
		idx_t col_pos = column_ids.size();
		for (idx_t i = 0; i < column_ids.size(); i++) {
			if (column_ids[i].GetPrimaryIndex() == entry.first) {
				col_pos = i;
				break;
			}
		}
		if (col_pos == column_ids.size()) {
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

	// aggregate programs (shared acceptance; grouped sums accumulate host-side
	// in 128-bit, so only per-row int64 bounds apply)
	vector<MlxSumProgram> programs;
	vector<LogicalType> agg_types;
	if (!BuildAggregatePrograms(context, get, projs, agg, estimated_rows, true, programs, agg_types)) {
		return false;
	}

	// scan projection: program columns + key columns; cache adds filter cols
	std::set<idx_t> agg_column_ids;
	CollectProgramColumnIds(programs, *get, agg_column_ids);
	auto storage_to_column_ids_idx = [&](idx_t storage_col) -> idx_t {
		for (idx_t i = 0; i < column_ids.size(); i++) {
			if (column_ids[i].GetPrimaryIndex() == storage_col) {
				return i;
			}
		}
		return column_ids.size();
	};
	for (auto storage_col : key_storage_cols) {
		auto pos = storage_to_column_ids_idx(NumericCast<idx_t>(storage_col));
		if (pos >= column_ids.size()) {
			return false;
		}
		agg_column_ids.insert(pos);
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
	vector<idx_t> cache_projection(scan_projection.begin(), scan_projection.end());
	for (auto storage_col : filter_cols) {
		auto column_ids_idx = storage_to_column_ids_idx(static_cast<idx_t>(storage_col));
		if (column_ids_idx >= column_ids.size()) {
			return false;
		}
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
	// keys: storage -> chunk positions
	for (auto &key : keys) {
		auto column_ids_idx = storage_to_column_ids_idx(NumericCast<idx_t>(key.scan_col));
		bool found = false;
		for (idx_t out = 0; out < scan_projection.size(); out++) {
			if (scan_projection[out] == column_ids_idx) {
				key.scan_col = NumericCast<int32_t>(out);
				found = true;
				break;
			}
		}
		if (!found) {
			return false;
		}
	}

	bool has_table_filters = !get->table_filters.filters.empty();

	vector<string> col_keys;
	string table_prefix;
	int64_t total_rows = 0;
	bool cached = false;
	auto table = get->GetTable();
	if (table) {
		table_prefix = table->ParentCatalog().GetName() + "." + table->ParentSchema().name + "." + table->name + "#";
		total_rows = NumericCast<int64_t>(table->GetStorage().GetTotalRows());
		for (auto pid : cache_projection) {
			col_keys.push_back(table_prefix + std::to_string(column_ids[pid].GetPrimaryIndex()));
		}
		for (auto &key : keys) {
			key.cache_key = col_keys[key.scan_col];
		}
		cached = duckdb_mlx::MlxCacheHas(col_keys, total_rows);
		if (cached) {
			for (auto &key : keys) {
				if (key.is_varchar &&
				    duckdb_mlx::MlxDictCard(key.cache_key, duckdb_mlx::MlxCachePopulation(key.cache_key)) <= 0) {
					cached = false;
					break;
				}
			}
			if (cached) {
				duckdb_mlx::MlxCacheBindDerivedPrograms(table_prefix, col_keys, programs);
			}
		}
	}

	get->projection_ids = std::move(scan_projection);
	bool skip_cache_populate = has_table_filters;

	auto mlx_op = make_uniq<MlxGroupedLogicalOperator>(agg.group_index, agg.aggregate_index, std::move(keys),
	                                                   std::move(programs), std::move(agg_types),
	                                                   std::move(cache_filter), std::move(col_keys),
	                                                   std::move(table_prefix), total_rows, cached,
	                                                   skip_cache_populate);
	if (parent_proj) {
		mlx_op->single_binding = true;
		mlx_op->output_index = parent_proj->table_index;
		mlx_op->output_map = std::move(output_map);
	}
	mlx_op->estimated_cardinality = agg.estimated_cardinality;
	mlx_op->has_estimated_cardinality = agg.has_estimated_cardinality;
	if (!cached) {
		// splice the projection chain out: programs and keys are expressed
		// over the scan's output columns
		if (!projs.empty()) {
			mlx_op->children.push_back(std::move(projs.back()->children[0]));
		} else {
			mlx_op->children.push_back(std::move(agg.children[0]));
		}
	}
	plan = std::move(mlx_op);
	duckdb_mlx::LogDebug(cached ? "MLX_GROUPBY serving from the GPU-resident cache (no scan)"
	                            : "MLX_GROUPBY intercepted a grouped aggregation");
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

	vector<MlxSumProgram> programs;
	vector<LogicalType> agg_types;
	if (!BuildAggregatePrograms(context, get, proj ? vector<LogicalProjection *> {proj.get()}
	                                               : vector<LogicalProjection *> {},
	                            agg, estimated_rows, false, programs, agg_types)) {
		return false;
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
	if (IntLaneCpuFast(programs)) {
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

static bool ColumnPinSupported(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::DATE:
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::DECIMAL:
		return true;
	default:
		return false;
	}
}

static int64_t DecimalUnitScaled(const LogicalType &type) {
	auto scale = DecimalType::GetScale(type);
	int64_t one = 1;
	for (uint8_t i = 0; i < scale; i++) {
		one *= 10;
	}
	return one;
}

static void MlxPinMaterializeDerived(const TableCatalogEntry &table, const string &table_prefix) {
	if (table.name != "lineitem") {
		return;
	}
	int32_t extendedprice = -1;
	int32_t discount = -1;
	int32_t tax_col = -1;
	int32_t shipdate = -1;
	int32_t returnflag = -1;
	int32_t linestatus = -1;
	int32_t quantity = -1;
	int64_t decimal_one = 100;
	auto &columns = table.GetColumns();
	for (idx_t col_idx = 0; col_idx < columns.LogicalColumnCount(); col_idx++) {
		auto &col = columns.GetColumn(LogicalIndex(col_idx));
		if (col.Name() == "l_extendedprice") {
			extendedprice = NumericCast<int32_t>(col_idx);
			decimal_one = DecimalUnitScaled(col.Type());
		} else if (col.Name() == "l_discount") {
			discount = NumericCast<int32_t>(col_idx);
		} else if (col.Name() == "l_tax") {
			tax_col = NumericCast<int32_t>(col_idx);
		} else if (col.Name() == "l_shipdate") {
			shipdate = NumericCast<int32_t>(col_idx);
		} else if (col.Name() == "l_returnflag") {
			returnflag = NumericCast<int32_t>(col_idx);
		} else if (col.Name() == "l_linestatus") {
			linestatus = NumericCast<int32_t>(col_idx);
		} else if (col.Name() == "l_quantity") {
			quantity = NumericCast<int32_t>(col_idx);
		}
	}
	if (extendedprice >= 0 && discount >= 0 && tax_col >= 0) {
		duckdb_mlx::MlxCacheMaterializeLineitemTpch(table_prefix, extendedprice, discount, tax_col, decimal_one);
	}
	if (shipdate >= 0 && returnflag >= 0 && linestatus >= 0 && quantity >= 0 && extendedprice >= 0 && discount >= 0) {
		duckdb_mlx::MlxCacheMaterializeLineitemQ1(table_prefix, shipdate, returnflag, linestatus, quantity, extendedprice,
		                                        discount);
	}
}

struct PinColumnBuffer {
	vector<float> values;
	vector<int64_t> ivalues;
	vector<uint8_t> valid;
	bool is_varchar = false;
	bool is_int_lane = false;
	unordered_map<string, int32_t> dict;
	vector<string> dict_strings;
};

static void AppendPinVarchar(UnifiedVectorFormat &fmt, idx_t count, vector<float> &values,
                             unordered_map<string, int32_t> &dict, vector<string> &decode) {
	auto strings = UnifiedVectorFormat::GetData<string_t>(fmt);
	auto base = values.size();
	values.resize(base + count);
	for (idx_t i = 0; i < count; i++) {
		auto idx = fmt.sel->get_index(i);
		if (!fmt.validity.RowIsValid(idx)) {
			values[base + i] = -1.0f;
			continue;
		}
		auto str = strings[idx].GetString();
		auto it = dict.find(str);
		int32_t code;
		if (it != dict.end()) {
			code = it->second;
		} else {
			code = NumericCast<int32_t>(decode.size());
			decode.push_back(str);
			dict.emplace(str, code);
		}
		values[base + i] = static_cast<float>(code);
	}
}

static void PinChunkColumns(DataChunk &chunk, vector<PinColumnBuffer> &columns, const vector<idx_t> &chunk_to_table) {
	for (idx_t col = 0; col < chunk.ColumnCount(); col++) {
		auto table_col = chunk_to_table[col];
		auto &buf = columns[table_col];
		UnifiedVectorFormat fmt;
		chunk.data[col].ToUnifiedFormat(chunk.size(), fmt);
		if (buf.is_varchar) {
			AppendPinVarchar(fmt, chunk.size(), buf.values, buf.dict, buf.dict_strings);
			continue;
		}
		if (!AppendLaneColumn(chunk.data[col].GetType(), fmt, chunk.size(), buf.values, buf.ivalues, buf.valid)) {
			throw InternalException("mlx_cache_pin: unsupported column type %s at execution time",
			                        chunk.data[col].GetType().ToString());
		}
	}
}

static void PinBuffersToMlxColumns(const vector<PinColumnBuffer> &columns, const vector<idx_t> &pinned_table_cols,
                                   vector<duckdb_mlx::MlxColumnData> &out) {
	for (auto table_col : pinned_table_cols) {
		auto &buf = columns[table_col];
		duckdb_mlx::MlxColumnData data;
		data.valid = buf.valid.empty() ? nullptr : buf.valid.data();
		if (!buf.ivalues.empty()) {
			data.ivalues = buf.ivalues.data();
		} else {
			data.values = buf.values.data();
		}
		out.push_back(data);
	}
}

MlxCachePinResult MlxCachePinTable(ClientContext &context, const string &table_name) {
	MlxCachePinResult result;
	auto qn = QualifiedName::Parse(table_name);
	CatalogEntryRetriever retriever(context);
	EntryLookupInfo lookup(CatalogType::TABLE_ENTRY, qn.name);
	optional_ptr<CatalogEntry> entry_ptr;
	if (!qn.catalog.empty() || !qn.schema.empty()) {
		entry_ptr = retriever.GetEntry(qn.catalog.empty() ? INVALID_CATALOG : qn.catalog,
		                               qn.schema.empty() ? DEFAULT_SCHEMA : qn.schema, lookup);
	} else {
		for (auto &path : retriever.GetSearchPath().Get()) {
			entry_ptr = retriever.GetEntry(path.catalog, path.schema, lookup, OnEntryNotFound::RETURN_NULL);
			if (entry_ptr) {
				break;
			}
		}
		if (!entry_ptr) {
			throw CatalogException("Table \"%s\" not found", qn.name);
		}
	}
	if (entry_ptr->type != CatalogType::TABLE_ENTRY) {
		throw InvalidInputException("mlx_cache_pin: '%s' is not a table", table_name);
	}
	auto &table = entry_ptr->Cast<TableCatalogEntry>();
	auto &storage = table.GetStorage();
	auto total_rows = NumericCast<int64_t>(storage.GetTotalRows());
	if (total_rows == 0) {
		return result;
	}

	string table_prefix =
	    table.ParentCatalog().GetName() + "." + table.schema.name + "." + table.name + "#";
	auto types = storage.GetTypes();
	vector<StorageIndex> storage_ids;
	vector<string> col_keys;
	vector<idx_t> pinned_table_cols;
	vector<bool> pin_col(types.size(), false);
	vector<PinColumnBuffer> columns(types.size());
	vector<LogicalType> scan_types;

	for (idx_t col_idx = 0; col_idx < types.size(); col_idx++) {
		if (!ColumnPinSupported(types[col_idx])) {
			continue;
		}
		storage_ids.emplace_back(col_idx);
		pinned_table_cols.push_back(col_idx);
		scan_types.push_back(types[col_idx]);
		col_keys.push_back(table_prefix + std::to_string(col_idx));
		pin_col[col_idx] = true;
		auto &buf = columns[col_idx];
		buf.is_varchar = types[col_idx].id() == LogicalTypeId::VARCHAR;
		buf.is_int_lane = types[col_idx].id() == LogicalTypeId::DECIMAL;
	}
	if (col_keys.empty()) {
		throw InvalidInputException("mlx_cache_pin: no pin-able columns in table '%s'", table_name);
	}

	if (duckdb_mlx::MlxCacheHas(col_keys, total_rows)) {
		result.rows = total_rows;
		result.columns = NumericCast<int64_t>(col_keys.size());
		result.already_resident = true;
		duckdb_mlx::MlxCacheFuseTable(table_prefix);
		MlxPinMaterializeDerived(table, table_prefix);
		duckdb_mlx::LogDebug("mlx_cache_pin: " + table.name + " already resident (" + std::to_string(total_rows) +
		                     " rows)");
		return result;
	}

	auto plan = duckdb_mlx::MlxCacheBeginPopulation(table_prefix, col_keys, total_rows);
	auto &transaction = DuckTransaction::Get(context, table.catalog);
	TableScanState scan_state;
	storage.InitializeScan(context, transaction, scan_state, storage_ids, nullptr);

	DataChunk chunk;
	chunk.Initialize(context, scan_types);
	vector<idx_t> chunk_to_table = pinned_table_cols;
	while (true) {
		chunk.Reset();
		storage.Scan(transaction, chunk, scan_state);
		if (chunk.size() == 0) {
			break;
		}
		PinChunkColumns(chunk, columns, chunk_to_table);
		vector<duckdb_mlx::MlxColumnData> mlx_cols;
		PinBuffersToMlxColumns(columns, pinned_table_cols, mlx_cols);
		duckdb_mlx::MlxCacheStoreSegment(plan.population, col_keys, plan.store_col, mlx_cols, chunk.size());
		for (auto table_col : pinned_table_cols) {
			columns[table_col].values.clear();
			columns[table_col].ivalues.clear();
			columns[table_col].valid.clear();
		}
	}

	duckdb_mlx::MlxCacheFuseTable(table_prefix);
	MlxPinMaterializeDerived(table, table_prefix);

	for (auto table_col : pinned_table_cols) {
		if (!columns[table_col].is_varchar) {
			continue;
		}
		duckdb_mlx::MlxDictInstall(table_prefix + std::to_string(table_col), plan.population,
		                           columns[table_col].dict_strings);
	}

	result.rows = total_rows;
	result.columns = NumericCast<int64_t>(col_keys.size());
	duckdb_mlx::LogDebug("mlx_cache_pin: pinned " + table.name + " (" + std::to_string(result.rows) + " rows, " +
	                     std::to_string(result.columns) + " columns)");
	return result;
}

void MlxCachePinTpch(ClientContext &context) {
	static const char *tables[] = {"customer", "lineitem", "nation", "orders", "part", "partsupp", "region",
	                               "supplier"};
	for (auto *name : tables) {
		MlxCachePinTable(context, name);
	}
}

void RegisterMlxOptimizer(DatabaseInstance &db) {
	OptimizerExtension ext;
	ext.optimize_function = MlxOptimizeFunction;
	OptimizerExtension::Register(DBConfig::GetConfig(db), ext);
}

} // namespace duckdb
