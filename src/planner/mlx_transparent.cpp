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
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "mlx_bridge.hpp"
#include "mlx_logger.hpp"

#include <set>

namespace duckdb {

using duckdb_mlx::MlxColumnData;
using duckdb_mlx::MlxExprOp;
using duckdb_mlx::MlxExprOpCode;
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
	                       vector<MlxSumProgram> programs_p, vector<string> col_keys_p, string table_prefix_p,
	                       bool cached_p)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
	      programs(std::move(programs_p)), col_keys(std::move(col_keys_p)), table_prefix(std::move(table_prefix_p)),
	      cached(cached_p) {
	}

	vector<MlxSumProgram> programs;
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
			gstate.results = duckdb_mlx::MlxSumExprsCached(col_keys, programs);
			duckdb_mlx::LogDebug("MLX_SUM populated the GPU cache for " + table_prefix);
			return SinkFinalizeType::READY;
		}
		gstate.results.assign(programs.size(), {0.0, 0});
		for (auto &segment : gstate.segments) {
			vector<MlxColumnData> cols;
			size_t row_count = segment.values.empty() ? 0 : segment.values[0].size();
			for (idx_t col = 0; col < segment.values.size(); col++) {
				cols.push_back(
				    {segment.values[col].data(), segment.valid[col].empty() ? nullptr : segment.valid[col].data()});
			}
			auto partial = duckdb_mlx::MlxSumExprs(cols, row_count, programs);
			for (idx_t i = 0; i < programs.size(); i++) {
				gstate.results[i].value += partial[i].value;
				gstate.results[i].valid_count += partial[i].valid_count;
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
			results = duckdb_mlx::MlxSumExprsCached(col_keys, programs);
		} else {
			results = sink_state->Cast<MlxSumGlobalSinkState>().results;
		}
		for (idx_t i = 0; i < results.size(); i++) {
			if (results[i].valid_count == 0) {
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
	MlxSumLogicalOperator(idx_t aggregate_index, vector<MlxSumProgram> programs_p, vector<string> col_keys_p,
	                      string table_prefix_p, bool cached_p)
	    : aggregate_index(aggregate_index), programs(std::move(programs_p)), col_keys(std::move(col_keys_p)),
	      table_prefix(std::move(table_prefix_p)), cached(cached_p) {
		estimated_cardinality = 1;
		has_estimated_cardinality = true;
	}

	idx_t aggregate_index;
	vector<MlxSumProgram> programs;
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
		types.clear();
		for (idx_t i = 0; i < programs.size(); i++) {
			types.push_back(LogicalType::DOUBLE);
		}
	}

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override {
		auto &op = planner.Make<MlxSumPhysicalOperator>(types, estimated_cardinality, std::move(programs),
		                                                std::move(col_keys), std::move(table_prefix), cached);
		if (!cached) {
			auto &child = planner.CreatePlan(*children[0]);
			op.children.push_back(child);
		}
		return op;
	}
};

//===--------------------------------------------------------------------===//
// Optimizer hook: match AGGREGATE(SUM...) <- [PROJECTION] <- GET and replace
//===--------------------------------------------------------------------===//
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

	// child shape: [PROJECTION ->] GET
	optional_ptr<LogicalProjection> proj;
	optional_ptr<LogicalGet> get;
	auto &child = *agg.children[0];
	if (child.type == LogicalOperatorType::LOGICAL_PROJECTION) {
		proj = &child.Cast<LogicalProjection>();
		if (proj->children[0]->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}
		get = &proj->children[0]->Cast<LogicalGet>();
	} else if (child.type == LogicalOperatorType::LOGICAL_GET) {
		get = &child.Cast<LogicalGet>();
	} else {
		return false;
	}
	if (get->function.name != "seq_scan" || !get->table_filters.filters.empty()) {
		return false;
	}
	auto estimated_rows =
	    get->has_estimated_cardinality ? get->estimated_cardinality : get->EstimateCardinality(context);
	if (estimated_rows < min_rows) {
		return false;
	}

	vector<MlxSumProgram> programs;
	for (auto &expr : agg.expressions) {
		if (expr->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			return false;
		}
		auto &aggr = expr->Cast<BoundAggregateExpression>();
		if (aggr.function.name != "sum" || aggr.IsDistinct() || aggr.filter || aggr.children.size() != 1 ||
		    aggr.return_type.id() != LogicalTypeId::DOUBLE) {
			return false;
		}
		MlxExprTranslator translator(*get, proj);
		if (!translator.Translate(*aggr.children[0])) {
			return false;
		}
		MlxSumProgram program;
		program.ops = std::move(translator.ops);
		program.null_cols.assign(translator.referenced_cols.begin(), translator.referenced_cols.end());
		programs.push_back(std::move(program));
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
		auto &column_ids = get->GetColumnIds();
		if (get->projection_ids.empty()) {
			for (auto &col : column_ids) {
				col_keys.push_back(table_prefix + std::to_string(col.GetPrimaryIndex()));
			}
		} else {
			for (auto pid : get->projection_ids) {
				col_keys.push_back(table_prefix + std::to_string(column_ids[pid].GetPrimaryIndex()));
			}
		}
		auto total_rows = NumericCast<int64_t>(table->GetStorage().GetTotalRows());
		cached = duckdb_mlx::MlxCacheHas(col_keys, total_rows);
	}

	auto mlx_op = make_uniq<MlxSumLogicalOperator>(agg.aggregate_index, std::move(programs), std::move(col_keys),
	                                               std::move(table_prefix), cached);
	if (!cached) {
		if (proj) {
			mlx_op->children.push_back(std::move(proj->children[0]));
		} else {
			mlx_op->children.push_back(std::move(agg.children[0]));
		}
	}
	plan = std::move(mlx_op);
	duckdb_mlx::LogDebug(cached ? "MLX_SUM serving from the GPU-resident cache (no scan)"
	                            : "MLX_SUM intercepted an ungrouped SUM aggregation");
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
