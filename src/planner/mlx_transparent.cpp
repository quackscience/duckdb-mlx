#include "mlx_transparent.hpp"

#include "duckdb.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/storage/data_table.hpp"
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

	static bool IsSupportedColumnType(const LogicalType &type) {
		switch (type.id()) {
		case LogicalTypeId::DOUBLE:
		case LogicalTypeId::FLOAT:
		case LogicalTypeId::BIGINT:
		case LogicalTypeId::INTEGER:
			return true;
		default:
			return false;
		}
	}

	bool Translate(const Expression &expr) {
		switch (expr.GetExpressionClass()) {
		case ExpressionClass::BOUND_COLUMN_REF: {
			auto &colref = expr.Cast<BoundColumnRefExpression>();
			if (proj && colref.binding.table_index == proj->table_index) {
				return Translate(*proj->expressions[colref.binding.column_index]);
			}
			if (colref.binding.table_index != get.table_index) {
				return false;
			}
			if (!IsSupportedColumnType(colref.return_type)) {
				return false;
			}
			auto col = NumericCast<int32_t>(colref.binding.column_index);
			ops.push_back({MlxExprOpCode::LOAD_COL, col, 0});
			referenced_cols.insert(col);
			return true;
		}
		case ExpressionClass::BOUND_CONSTANT: {
			auto &constant = expr.Cast<BoundConstantExpression>();
			if (constant.value.IsNull() || !constant.value.type().IsNumeric()) {
				return false;
			}
			ops.push_back({MlxExprOpCode::CONST_VAL, 0, constant.value.GetValue<double>()});
			return true;
		}
		case ExpressionClass::BOUND_CAST: {
			auto &cast = expr.Cast<BoundCastExpression>();
			if (cast.try_cast) {
				return false;
			}
			// numeric -> DOUBLE/FLOAT casts are implicit in the fp32 pipeline
			if (cast.return_type.id() != LogicalTypeId::DOUBLE && cast.return_type.id() != LogicalTypeId::FLOAT) {
				return false;
			}
			if (!cast.child->return_type.IsNumeric()) {
				return false;
			}
			return Translate(*cast.child);
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
			if (!Translate(*comparison.left) || !Translate(*comparison.right)) {
				return false;
			}
			ops.push_back({code, 0, 0});
			return true;
		}
		case ExpressionClass::BOUND_CONJUNCTION: {
			auto &conjunction = expr.Cast<BoundConjunctionExpression>();
			auto code =
			    expr.GetExpressionType() == ExpressionType::CONJUNCTION_AND ? MlxExprOpCode::AND : MlxExprOpCode::OR;
			if (!Translate(*conjunction.children[0])) {
				return false;
			}
			for (idx_t i = 1; i < conjunction.children.size(); i++) {
				if (!Translate(*conjunction.children[i])) {
					return false;
				}
				ops.push_back({code, 0, 0});
			}
			return true;
		}
		case ExpressionClass::BOUND_OPERATOR: {
			auto &op = expr.Cast<BoundOperatorExpression>();
			if (expr.GetExpressionType() != ExpressionType::OPERATOR_NOT || op.children.size() != 1) {
				return false;
			}
			if (!Translate(*op.children[0])) {
				return false;
			}
			ops.push_back({MlxExprOpCode::NOT, 0, 0});
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
				if (!Translate(*function.children[0]) || !Translate(*function.children[1])) {
					return false;
				}
				ops.push_back({code, 0, 0});
				return true;
			}
			if (function.children.size() == 1) {
				MlxExprOpCode code;
				if (name == "-") {
					code = MlxExprOpCode::NEGATE;
				} else if (name == "sin") {
					code = MlxExprOpCode::SIN;
				} else if (name == "cos") {
					code = MlxExprOpCode::COS;
				} else if (name == "sqrt") {
					code = MlxExprOpCode::SQRT;
				} else if (name == "abs") {
					code = MlxExprOpCode::ABS;
				} else {
					return false;
				}
				if (!Translate(*function.children[0])) {
					return false;
				}
				ops.push_back({code, 0, 0});
				return true;
			}
			return false;
		}
		default:
			return false;
		}
	}
};

//===--------------------------------------------------------------------===//
// Physical operator: sinks child chunks into unified-memory column buffers,
// evaluates all SUM programs in one GPU graph at Finalize, emits one row.
//===--------------------------------------------------------------------===//
//! One thread's buffered slice of the input; segments are evaluated on the
//! GPU independently and their partial sums accumulated (SUM is decomposable),
//! so combining never re-copies the data.
struct MlxSumSegment {
	vector<vector<float>> values;
	vector<vector<uint8_t>> valid;
};

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
	                       vector<MlxSumProgram> programs_p, MlxFilter filter_p, vector<string> col_keys_p,
	                       string table_prefix_p, bool cached_p)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
	      programs(std::move(programs_p)), filter(std::move(filter_p)), col_keys(std::move(col_keys_p)),
	      table_prefix(std::move(table_prefix_p)), cached(cached_p) {
	}

	vector<MlxSumProgram> programs;
	MlxFilter filter;
	//! GPU cache keys of the child's output columns; empty when caching is off
	vector<string> col_keys;
	string table_prefix;
	//! When true the operator has no child: it is a pure source reading the
	//! GPU-resident column cache (no table scan at all)
	bool cached;

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
			segment.valid.resize(chunk.ColumnCount()); // stays empty per column until a NULL appears
		}
		auto count = chunk.size();
		for (idx_t col = 0; col < chunk.ColumnCount(); col++) {
			UnifiedVectorFormat fmt;
			chunk.data[col].ToUnifiedFormat(count, fmt);
			auto &values = segment.values[col];
			auto &valid = segment.valid[col];
			switch (chunk.data[col].GetType().id()) {
			case LogicalTypeId::DOUBLE:
				AppendColumn<double>(fmt, count, values, valid);
				break;
			case LogicalTypeId::FLOAT:
				AppendColumn<float>(fmt, count, values, valid);
				break;
			case LogicalTypeId::BIGINT:
				AppendColumn<int64_t>(fmt, count, values, valid);
				break;
			case LogicalTypeId::INTEGER:
				AppendColumn<int32_t>(fmt, count, values, valid);
				break;
			default:
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
		if (!col_keys.empty()) {
			// populate the GPU column cache (new population: drops the table's
			// previous columns so multi-column programs stay row-aligned),
			// then evaluate from the cache — later queries skip the scan
			auto population = duckdb_mlx::MlxCacheBeginPopulation(table_prefix);
			for (auto &segment : gstate.segments) {
				vector<MlxColumnData> cols;
				size_t row_count = segment.values.empty() ? 0 : segment.values[0].size();
				for (idx_t col = 0; col < segment.values.size(); col++) {
					cols.push_back(
					    {segment.values[col].data(), segment.valid[col].empty() ? nullptr : segment.valid[col].data()});
				}
				duckdb_mlx::MlxCacheStoreSegment(population, col_keys, cols, row_count);
			}
			gstate.results = duckdb_mlx::MlxSumExprsCached(col_keys, programs, filter);
			duckdb_mlx::LogDebug("MLX_SUM populated the GPU cache for " + table_prefix);
			return SinkFinalizeType::READY;
		}
		gstate.results.clear();
		for (auto &program : programs) {
			switch (program.kind) {
			case MlxAggKind::MIN:
				gstate.results.push_back({std::numeric_limits<double>::infinity(), 0});
				break;
			case MlxAggKind::MAX:
				gstate.results.push_back({-std::numeric_limits<double>::infinity(), 0});
				break;
			default:
				gstate.results.push_back({0.0, 0});
				break;
			}
		}
		for (auto &segment : gstate.segments) {
			vector<MlxColumnData> cols;
			size_t row_count = segment.values.empty() ? 0 : segment.values[0].size();
			for (idx_t col = 0; col < segment.values.size(); col++) {
				cols.push_back(
				    {segment.values[col].data(), segment.valid[col].empty() ? nullptr : segment.valid[col].data()});
			}
			auto partial = duckdb_mlx::MlxSumExprs(cols, row_count, programs, filter);
			for (idx_t i = 0; i < programs.size(); i++) {
				switch (programs[i].kind) {
				case MlxAggKind::MIN:
					gstate.results[i].value = MinValue(gstate.results[i].value, partial[i].value);
					break;
				case MlxAggKind::MAX:
					gstate.results[i].value = MaxValue(gstate.results[i].value, partial[i].value);
					break;
				case MlxAggKind::AVG:
					// re-weight the per-segment average into a running sum
					gstate.results[i].value += partial[i].value * static_cast<double>(partial[i].valid_count);
					break;
				default:
					gstate.results[i].value += partial[i].value;
					break;
				}
				gstate.results[i].valid_count += partial[i].valid_count;
			}
		}
		for (idx_t i = 0; i < programs.size(); i++) {
			if (programs[i].kind == MlxAggKind::AVG && gstate.results[i].valid_count > 0) {
				gstate.results[i].value /= static_cast<double>(gstate.results[i].valid_count);
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
			results = duckdb_mlx::MlxSumExprsCached(col_keys, programs, filter);
		} else {
			results = sink_state->Cast<MlxSumGlobalSinkState>().results;
		}
		for (idx_t i = 0; i < results.size(); i++) {
			if (programs[i].kind == MlxAggKind::COUNT || programs[i].kind == MlxAggKind::COUNT_STAR) {
				FlatVector::GetData<int64_t>(chunk.data[i])[0] = results[i].valid_count;
			} else if (results[i].valid_count == 0) {
				FlatVector::Validity(chunk.data[i]).SetInvalid(0);
			} else {
				FlatVector::GetData<double>(chunk.data[i])[0] = results[i].value;
			}
		}
		chunk.SetCardinality(1);
		source_state.done = true;
		return SourceResultType::FINISHED;
	}

private:
	template <class T>
	static void AppendColumn(UnifiedVectorFormat &fmt, idx_t count, vector<float> &values, vector<uint8_t> &valid) {
		auto data = UnifiedVectorFormat::GetData<T>(fmt);
		auto base = values.size();
		values.resize(base + count);
		if (!fmt.sel->IsSet() && fmt.validity.AllValid()) {
			// flat, NULL-free fast path: tight convert loop, auto-vectorized;
			// the validity buffer is only materialized once a NULL appears
			for (idx_t i = 0; i < count; i++) {
				values[base + i] = static_cast<float>(data[i]);
			}
			if (!valid.empty()) {
				valid.resize(base + count, 1);
			}
			return;
		}
		bool chunk_has_nulls = !fmt.validity.AllValid();
		if (chunk_has_nulls || !valid.empty()) {
			// backfills rows appended before the first NULL with 1s
			valid.resize(base + count, 1);
		}
		for (idx_t i = 0; i < count; i++) {
			auto idx = fmt.sel->get_index(i);
			bool row_valid = fmt.validity.RowIsValid(idx);
			values[base + i] = row_valid ? static_cast<float>(data[idx]) : 0.0f;
			if (!row_valid) {
				valid[base + i] = 0;
			}
		}
	}
};

//===--------------------------------------------------------------------===//
// Logical operator
//===--------------------------------------------------------------------===//
class MlxSumLogicalOperator : public LogicalExtensionOperator {
public:
	MlxSumLogicalOperator(idx_t aggregate_index, vector<MlxSumProgram> programs_p, vector<LogicalType> agg_types_p,
	                      MlxFilter filter_p, vector<string> col_keys_p, string table_prefix_p, bool cached_p)
	    : aggregate_index(aggregate_index), programs(std::move(programs_p)), agg_types(std::move(agg_types_p)),
	      filter(std::move(filter_p)), col_keys(std::move(col_keys_p)), table_prefix(std::move(table_prefix_p)),
	      cached(cached_p) {
		estimated_cardinality = 1;
		has_estimated_cardinality = true;
	}

	idx_t aggregate_index;
	vector<MlxSumProgram> programs;
	vector<LogicalType> agg_types;
	MlxFilter filter;
	vector<string> col_keys;
	string table_prefix;
	bool cached;

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
		auto &op =
		    planner.Make<MlxSumPhysicalOperator>(types, estimated_cardinality, std::move(programs), std::move(filter),
		                                         std::move(col_keys), std::move(table_prefix), cached);
		if (!cached) {
			auto &child = planner.CreatePlan(*children[0]);
			op.children.push_back(child);
		}
		return op;
	}
};

//===--------------------------------------------------------------------===//
// Optimizer hook: match AGGREGATE <- [PROJECTION] <- [FILTER] <- GET and
// replace. WHERE predicates (residual filter nodes and pushed-down table
// filters) become GPU row masks.
//===--------------------------------------------------------------------===//

//! Translates a pushed-down TableFilter tree into predicate IR. `col_pos` is
//! the column's position in the GET's column_ids (the space LOAD_COL uses
//! before remapping); `storage_idx` indexes the table schema for type checks.
static bool TranslateTableFilter(const TableFilter &table_filter, idx_t col_pos, idx_t storage_idx,
                                 const LogicalGet &get, vector<MlxExprOp> &ops, std::set<int32_t> &referenced) {
	switch (table_filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = table_filter.Cast<ConstantFilter>();
		if (constant_filter.constant.IsNull() || !constant_filter.constant.type().IsNumeric()) {
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
		ops.push_back({MlxExprOpCode::LOAD_COL, NumericCast<int32_t>(col_pos), 0});
		ops.push_back({MlxExprOpCode::CONST_VAL, 0, constant_filter.constant.GetValue<double>()});
		ops.push_back({code, 0, 0});
		referenced.insert(NumericCast<int32_t>(col_pos));
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
		referenced.insert(NumericCast<int32_t>(col_pos));
		return true;
	default:
		return false;
	}
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

	// WHERE predicate: residual filter expressions AND pushed-down table
	// filters, folded into one IR program over the GET's bound columns
	MlxFilter filter;
	std::set<int32_t> filter_cols;
	if (filter_op) {
		bool first = true;
		for (auto &expr : filter_op->expressions) {
			MlxExprTranslator translator(*get, nullptr);
			if (!translator.Translate(*expr)) {
				return false;
			}
			filter.ops.insert(filter.ops.end(), translator.ops.begin(), translator.ops.end());
			filter_cols.insert(translator.referenced_cols.begin(), translator.referenced_cols.end());
			if (!first) {
				filter.ops.push_back({MlxExprOpCode::AND, 0, 0});
			}
			first = false;
		}
	}
	for (auto &entry : get->table_filters.filters) {
		// table filter keys are storage column indexes; map back to the
		// column's position in column_ids
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
			bool had_ops = !filter.ops.empty();
			filter.ops.insert(filter.ops.end(), ops.begin(), ops.end());
			if (had_ops) {
				filter.ops.push_back({MlxExprOpCode::AND, 0, 0});
			}
		}
	}
	filter.null_cols.assign(filter_cols.begin(), filter_cols.end());

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
		} else if (aggr.children.size() == 1 && aggr.return_type.id() == LogicalTypeId::DOUBLE) {
			if (name == "sum") {
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
			if (!translator.Translate(*aggr.children[0])) {
				return false;
			}
			program.ops = std::move(translator.ops);
			program.null_cols.assign(translator.referenced_cols.begin(), translator.referenced_cols.end());
		}
		agg_types.push_back(program.kind == MlxAggKind::COUNT || program.kind == MlxAggKind::COUNT_STAR
		                        ? LogicalType::BIGINT
		                        : LogicalType::DOUBLE);
		programs.push_back(std::move(program));
	}

	// filter-only columns are pruned from the scan's projection; since their
	// predicates now run as GPU masks, extend the projection to emit them
	auto &column_ids = get->GetColumnIds();
	auto extended_projection = get->projection_ids;
	if (!extended_projection.empty()) {
		for (auto col : filter_cols) {
			bool present = false;
			for (auto pid : extended_projection) {
				if (NumericCast<int32_t>(pid) == col) {
					present = true;
					break;
				}
			}
			if (!present) {
				extended_projection.push_back(NumericCast<idx_t>(col));
			}
		}
	}

	// remap bound-column indexes to the GET's output chunk positions
	vector<int32_t> output_map(column_ids.size(), -1);
	if (extended_projection.empty()) {
		for (idx_t i = 0; i < column_ids.size(); i++) {
			output_map[i] = NumericCast<int32_t>(i);
		}
	} else {
		for (idx_t out = 0; out < extended_projection.size(); out++) {
			output_map[extended_projection[out]] = NumericCast<int32_t>(out);
		}
	}
	auto remap = [&](std::vector<MlxExprOp> &ops, std::vector<int32_t> &null_cols) {
		for (auto &op : ops) {
			if (op.code == MlxExprOpCode::LOAD_COL) {
				if (output_map[op.col] < 0) {
					return false;
				}
				op.col = output_map[op.col];
			}
		}
		for (auto &col : null_cols) {
			if (output_map[col] < 0) {
				return false;
			}
			col = output_map[col];
		}
		return true;
	};
	if (!remap(filter.ops, filter.null_cols)) {
		return false;
	}
	for (auto &program : programs) {
		if (!remap(program.ops, program.null_cols)) {
			return false;
		}
	}

	// GPU column cache identity: catalog.schema.table plus each output
	// column's storage id. A row-count match against current storage decides
	// whether the cached population is served (GQE-style resident table).
	vector<string> col_keys;
	string table_prefix;
	bool cached = false;
	auto table = get->GetTable();
	if (table) {
		table_prefix = table->ParentCatalog().GetName() + "." + table->ParentSchema().name + "." + table->name + "#";
		if (extended_projection.empty()) {
			for (auto &col : column_ids) {
				col_keys.push_back(table_prefix + std::to_string(col.GetPrimaryIndex()));
			}
		} else {
			for (auto pid : extended_projection) {
				col_keys.push_back(table_prefix + std::to_string(column_ids[pid].GetPrimaryIndex()));
			}
		}
		auto total_rows = NumericCast<int64_t>(table->GetStorage().GetTotalRows());
		cached = duckdb_mlx::MlxCacheHas(col_keys, total_rows);
	}

	// commit the plan mutations: the WHERE predicate is fully translated to a
	// GPU mask, so the scan emits unfiltered rows (the cache must hold the
	// whole table) including the filter-only columns
	get->projection_ids = std::move(extended_projection);
	get->table_filters.filters.clear();

	auto mlx_op =
	    make_uniq<MlxSumLogicalOperator>(agg.aggregate_index, std::move(programs), std::move(agg_types),
	                                     std::move(filter), std::move(col_keys), std::move(table_prefix), cached);
	if (!cached) {
		if (filter_op) {
			mlx_op->children.push_back(std::move(filter_op->children[0]));
		} else if (proj) {
			mlx_op->children.push_back(std::move(proj->children[0]));
		} else {
			mlx_op->children.push_back(std::move(agg.children[0]));
		}
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
	TryInterceptAggregate(context, plan, min_rows);
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
