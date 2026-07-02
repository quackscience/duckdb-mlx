#include "mlx_bridge.hpp"

// This file must not include any DuckDB header (see mlx_bridge.hpp).
#include <mlx/mlx.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mx = mlx::core;

namespace duckdb_mlx {

namespace {
std::mutex vss_mutex;
// Process-lifetime GPU-resident cache of L2-normalized embedding matrices.
std::unordered_map<std::string, mx::array> &VssStore() {
	static std::unordered_map<std::string, mx::array> store;
	return store;
}
} // namespace

int64_t MlxVssPin(const std::string &name, const float *data, int64_t n, int64_t dim, bool half) {
	mx::array m(data, mx::Shape {static_cast<int>(n), static_cast<int>(dim)}, mx::float32);
	auto norms = mx::sqrt(mx::sum(mx::square(m), {1}, true));
	auto normalized = mx::divide(m, mx::maximum(norms, mx::array(1e-12f)));
	if (half) {
		normalized = mx::astype(normalized, mx::float16);
	}
	normalized.eval();
	std::lock_guard<std::mutex> guard(vss_mutex);
	VssStore().erase(name);
	VssStore().emplace(name, std::move(normalized));
	return n;
}

std::vector<MlxVssMatch> MlxVssSearch(const std::string &name, const float *query, int64_t dim, int64_t k) {
	mx::array m = [&] {
		std::lock_guard<std::mutex> guard(vss_mutex);
		auto it = VssStore().find(name);
		if (it == VssStore().end()) {
			throw std::runtime_error("no pinned matrix named '" + name + "' — call mlx_vss_pin first");
		}
		return it->second;
	}();
	if (m.shape(1) != dim) {
		throw std::runtime_error("query dimension " + std::to_string(dim) + " does not match pinned dimension " +
		                         std::to_string(m.shape(1)));
	}
	auto n = m.shape(0);
	k = std::min<int64_t>(k, n);
	if (k <= 0) {
		return {};
	}

	mx::array q(query, mx::Shape {static_cast<int>(dim)}, mx::float32);
	auto qn = mx::divide(q, mx::maximum(mx::sqrt(mx::sum(mx::square(q))), mx::array(1e-12f)));
	auto scores = mx::astype(mx::matmul(m, mx::astype(qn, m.dtype())), mx::float32);
	// argpartition is O(N); the k selected rows are ordered on the host
	auto part = mx::argpartition(mx::negative(scores), static_cast<int>(k - 1), 0);
	mx::eval({part, scores});

	auto part_ptr = part.data<uint32_t>();
	auto scores_ptr = scores.data<float>();
	std::vector<MlxVssMatch> matches;
	matches.reserve(k);
	for (int64_t i = 0; i < k; i++) {
		matches.push_back({part_ptr[i], scores_ptr[part_ptr[i]]});
	}
	std::sort(matches.begin(), matches.end(), [](const MlxVssMatch &a, const MlxVssMatch &b) {
		return a.score != b.score ? a.score > b.score : a.index < b.index;
	});
	return matches;
}

std::string MlxVersion() {
	return mx::version();
}

std::string MlxSelftest() {
	try {
		if (!mx::metal::is_available()) {
			return "metal unavailable";
		}
		if (mx::default_device().type != mx::Device::gpu) {
			return "default device is not gpu";
		}
		// int64 reduction: sum(0..99999) == 4999950000
		std::vector<int64_t> seq(100000);
		std::iota(seq.begin(), seq.end(), 0);
		mx::array a(seq.begin(), mx::Shape {static_cast<int>(seq.size())}, mx::int64);
		auto sum = mx::sum(a).item<int64_t>();
		if (sum != 4999950000LL) {
			return "int64 sum mismatch: got " + std::to_string(sum);
		}
		// float32 elementwise + reduction: sum(2*2 x 1024) == 4096
		auto b = mx::full({1024}, 2.0f);
		auto sq = mx::sum(mx::multiply(b, b)).item<float>();
		if (sq != 4096.0f) {
			return "float32 multiply/sum mismatch: got " + std::to_string(sq);
		}
		return "ok";
	} catch (const std::exception &ex) {
		return std::string("exception: ") + ex.what();
	}
}

std::vector<MlxVssBatchMatch> MlxVssSearchBatch(const std::string &name, const float *queries, int64_t q, int64_t dim,
                                                int64_t k) {
	mx::array m = [&] {
		std::lock_guard<std::mutex> guard(vss_mutex);
		auto it = VssStore().find(name);
		if (it == VssStore().end()) {
			throw std::runtime_error("no pinned matrix named '" + name + "' — call mlx_vss_pin first");
		}
		return it->second;
	}();
	if (m.shape(1) != dim) {
		throw std::runtime_error("query dimension " + std::to_string(dim) + " does not match pinned dimension " +
		                         std::to_string(m.shape(1)));
	}
	auto n = m.shape(0);
	k = std::min<int64_t>(k, n);
	if (k <= 0 || q <= 0) {
		return {};
	}

	mx::array qm(queries, mx::Shape {static_cast<int>(q), static_cast<int>(dim)}, mx::float32);
	auto qnorms = mx::sqrt(mx::sum(mx::square(qm), {1}, true));
	auto qn = mx::divide(qm, mx::maximum(qnorms, mx::array(1e-12f)));
	auto scores = mx::astype(mx::matmul(mx::astype(qn, m.dtype()), mx::transpose(m)), mx::float32); // (Q, N)
	// argpartition is O(N) per row; each row's k selected entries are ordered
	// on the host
	auto part = mx::argpartition(mx::negative(scores), static_cast<int>(k - 1), 1);
	mx::eval({part, scores});

	auto part_ptr = part.data<uint32_t>();
	auto scores_ptr = scores.data<float>();
	std::vector<MlxVssBatchMatch> matches;
	matches.reserve(q * k);
	std::vector<MlxVssMatch> row;
	for (int64_t qi = 0; qi < q; qi++) {
		row.clear();
		for (int64_t i = 0; i < k; i++) {
			auto idx = part_ptr[qi * n + i];
			row.push_back({idx, scores_ptr[qi * n + idx]});
		}
		std::sort(row.begin(), row.end(), [](const MlxVssMatch &a, const MlxVssMatch &b) {
			return a.score != b.score ? a.score > b.score : a.index < b.index;
		});
		for (auto &match : row) {
			matches.push_back({qi, match.index, match.score});
		}
	}
	return matches;
}

namespace {

//! One row-aligned slice of the input: per-column value arrays plus optional
//! validity arrays.
struct MlxSegment {
	std::vector<mx::array> cols;
	std::vector<std::optional<mx::array>> valids;
	int64_t count = 0;
};

MlxSegment SegmentFromHost(const std::vector<MlxColumnData> &cols, size_t count) {
	MlxSegment segment;
	segment.count = static_cast<int64_t>(count);
	auto shape = mx::Shape {static_cast<int>(count)};
	for (auto &col : cols) {
		segment.cols.emplace_back(mx::array(col.values, shape, mx::float32));
		if (col.valid) {
			segment.valids.emplace_back(mx::array(col.valid, shape, mx::uint8));
		} else {
			segment.valids.emplace_back(std::nullopt);
		}
	}
	return segment;
}

//! Columns of `cols` that carry a validity array in this segment.
std::vector<int32_t> MaskedCols(const MlxSegment &segment, const std::vector<int32_t> &cols) {
	std::vector<int32_t> masked;
	for (auto col : cols) {
		if (segment.valids[col].has_value()) {
			masked.push_back(col);
		}
	}
	return masked;
}

//! Whether this program needs a row mask in this segment (WHERE filter or
//! NULLs in referenced columns).
bool SegmentHasMask(const MlxSegment &segment, const MlxFilter &filter, const MlxSumProgram &program) {
	return !filter.ops.empty() || !MaskedCols(segment, filter.null_cols).empty() ||
	       !MaskedCols(segment, program.null_cols).empty();
}

bool HasValueGraph(MlxAggKind kind) {
	return kind != MlxAggKind::COUNT && kind != MlxAggKind::COUNT_STAR;
}

//! Evaluates a postfix program over the segment's columns.
mx::array EvalOps(const MlxSegment &segment, const std::vector<MlxExprOp> &ops) {
	std::vector<mx::array> stack;
	for (auto &op : ops) {
		switch (op.code) {
		case MlxExprOpCode::LOAD_COL:
			stack.push_back(segment.cols[op.col]);
			break;
		case MlxExprOpCode::CONST_VAL:
			stack.push_back(mx::array(static_cast<float>(op.value)));
			break;
		case MlxExprOpCode::NEGATE:
		case MlxExprOpCode::SIN:
		case MlxExprOpCode::COS:
		case MlxExprOpCode::SQRT:
		case MlxExprOpCode::ABS:
		case MlxExprOpCode::NOT: {
			auto a = stack.back();
			stack.pop_back();
			switch (op.code) {
			case MlxExprOpCode::NEGATE:
				stack.push_back(mx::negative(a));
				break;
			case MlxExprOpCode::SIN:
				stack.push_back(mx::sin(a));
				break;
			case MlxExprOpCode::COS:
				stack.push_back(mx::cos(a));
				break;
			case MlxExprOpCode::SQRT:
				stack.push_back(mx::sqrt(a));
				break;
			case MlxExprOpCode::ABS:
				stack.push_back(mx::abs(a));
				break;
			default:
				stack.push_back(mx::logical_not(a));
				break;
			}
			break;
		}
		default: {
			auto b = stack.back();
			stack.pop_back();
			auto a = stack.back();
			stack.pop_back();
			switch (op.code) {
			case MlxExprOpCode::ADD:
				stack.push_back(mx::add(a, b));
				break;
			case MlxExprOpCode::SUB:
				stack.push_back(mx::subtract(a, b));
				break;
			case MlxExprOpCode::MUL:
				stack.push_back(mx::multiply(a, b));
				break;
			case MlxExprOpCode::DIV:
				stack.push_back(mx::divide(a, b));
				break;
			case MlxExprOpCode::CMP_LT:
				stack.push_back(mx::less(a, b));
				break;
			case MlxExprOpCode::CMP_LE:
				stack.push_back(mx::less_equal(a, b));
				break;
			case MlxExprOpCode::CMP_GT:
				stack.push_back(mx::greater(a, b));
				break;
			case MlxExprOpCode::CMP_GE:
				stack.push_back(mx::greater_equal(a, b));
				break;
			case MlxExprOpCode::CMP_EQ:
				stack.push_back(mx::equal(a, b));
				break;
			case MlxExprOpCode::CMP_NE:
				stack.push_back(mx::not_equal(a, b));
				break;
			default:
				stack.push_back(op.code == MlxExprOpCode::AND ? mx::logical_and(a, b) : mx::logical_or(a, b));
				break;
			}
			break;
		}
		}
	}
	return stack.back();
}

//! AND of the validity arrays of `cols` (empty when none carry NULLs).
std::optional<mx::array> NullMask(const MlxSegment &segment, const std::vector<int32_t> &cols) {
	std::optional<mx::array> mask;
	for (auto col : MaskedCols(segment, cols)) {
		auto m = mx::astype(*segment.valids[col], mx::bool_);
		mask = mask.has_value() ? mx::logical_and(*mask, m) : m;
	}
	return mask;
}

std::optional<mx::array> CombineMasks(const std::optional<mx::array> &a, const std::optional<mx::array> &b) {
	if (!a.has_value()) {
		return b;
	}
	if (!b.has_value()) {
		return a;
	}
	return mx::logical_and(*a, *b);
}

//! Appends the lazy graphs of every aggregate over one segment: a value graph
//! (kind-dependent) and, when a row mask exists, a valid-count graph. Layout
//! per program: [value?][count?]. Constant-valued outputs must never enter a
//! compiled graph (mx::compile dedupes them, corrupting the output mapping);
//! maskless counts are accumulated host-side instead.
void BuildSumGraphs(const MlxSegment &segment, const std::vector<MlxSumProgram> &programs, const MlxFilter &filter,
                    std::vector<mx::array> &outs) {
	std::optional<mx::array> filter_mask;
	if (!filter.ops.empty()) {
		filter_mask = EvalOps(segment, filter.ops);
	}
	filter_mask = CombineMasks(filter_mask, NullMask(segment, filter.null_cols));

	for (auto &program : programs) {
		auto mask = CombineMasks(filter_mask, NullMask(segment, program.null_cols));
		if (HasValueGraph(program.kind)) {
			auto expr = EvalOps(segment, program.ops);
			switch (program.kind) {
			case MlxAggKind::MIN:
				outs.push_back(mask.has_value()
				                   ? mx::min(mx::where(*mask, expr, mx::array(std::numeric_limits<float>::infinity())))
				                   : mx::min(expr));
				break;
			case MlxAggKind::MAX:
				outs.push_back(mask.has_value()
				                   ? mx::max(mx::where(*mask, expr, mx::array(-std::numeric_limits<float>::infinity())))
				                   : mx::max(expr));
				break;
			default: // SUM / AVG
				outs.push_back(mask.has_value() ? mx::sum(mx::where(*mask, expr, mx::array(0.0f))) : mx::sum(expr));
				break;
			}
		}
		if (mask.has_value()) {
			outs.push_back(mx::sum(mx::astype(*mask, mx::int64)));
		}
	}
}

std::vector<MlxSumResult> EvalSegments(const std::vector<MlxSegment> &segments,
                                       const std::vector<MlxSumProgram> &programs, const MlxFilter &filter) {
	std::vector<MlxSumResult> results;
	for (auto &program : programs) {
		switch (program.kind) {
		case MlxAggKind::MIN:
			results.push_back({std::numeric_limits<double>::infinity(), 0});
			break;
		case MlxAggKind::MAX:
			results.push_back({-std::numeric_limits<double>::infinity(), 0});
			break;
		default:
			results.push_back({0.0, 0});
			break;
		}
	}

	std::vector<mx::array> all;
	std::vector<const MlxSegment *> evaluated;
	for (auto &segment : segments) {
		if (segment.count == 0) {
			continue;
		}
		// mx::compile fuses each segment's whole expression forest into a few
		// kernels instead of materializing every intermediate array
		std::vector<mx::array> inputs = segment.cols;
		std::vector<int32_t> valid_slots(segment.valids.size(), -1);
		for (size_t i = 0; i < segment.valids.size(); i++) {
			if (segment.valids[i].has_value()) {
				valid_slots[i] = static_cast<int32_t>(inputs.size());
				inputs.push_back(*segment.valids[i]);
			}
		}
		auto ncols = segment.cols.size();
		auto count = segment.count;
		auto fn = [&programs, &filter, ncols, &valid_slots, count](const std::vector<mx::array> &ins) {
			MlxSegment seg;
			seg.count = count;
			seg.cols.assign(ins.begin(), ins.begin() + ncols);
			for (size_t i = 0; i < ncols; i++) {
				if (valid_slots[i] >= 0) {
					seg.valids.emplace_back(ins[valid_slots[i]]);
				} else {
					seg.valids.emplace_back(std::nullopt);
				}
			}
			std::vector<mx::array> outs;
			BuildSumGraphs(seg, programs, filter, outs);
			return outs;
		};
		auto outs = mx::compile(fn)(inputs);
		all.insert(all.end(), outs.begin(), outs.end());
		evaluated.push_back(&segment);
	}
	mx::eval(all);

	size_t cursor = 0;
	for (auto segment : evaluated) {
		for (size_t p = 0; p < programs.size(); p++) {
			auto &program = programs[p];
			double value = 0;
			if (HasValueGraph(program.kind)) {
				value = static_cast<double>(all[cursor++].item<float>());
			}
			int64_t count = SegmentHasMask(*segment, filter, program) ? all[cursor++].item<int64_t>() : segment->count;
			switch (program.kind) {
			case MlxAggKind::MIN:
				results[p].value = std::min(results[p].value, value);
				break;
			case MlxAggKind::MAX:
				results[p].value = std::max(results[p].value, value);
				break;
			case MlxAggKind::COUNT:
			case MlxAggKind::COUNT_STAR:
				break;
			default: // SUM / AVG
				results[p].value += value;
				break;
			}
			results[p].valid_count += count;
		}
	}
	for (size_t p = 0; p < programs.size(); p++) {
		if (programs[p].kind == MlxAggKind::AVG && results[p].valid_count > 0) {
			results[p].value /= static_cast<double>(results[p].valid_count);
		}
	}
	return results;
}

std::mutex cache_mutex;
std::atomic<int64_t> cache_population_counter {0};

struct MlxCachedColumn {
	std::vector<mx::array> segments;
	std::vector<std::optional<mx::array>> valids;
	int64_t rows = 0;
	int64_t population = 0;
};

std::unordered_map<std::string, MlxCachedColumn> &CacheStore() {
	static std::unordered_map<std::string, MlxCachedColumn> store;
	return store;
}

} // namespace

std::vector<MlxSumResult> MlxSumExprs(const std::vector<MlxColumnData> &cols, size_t count,
                                      const std::vector<MlxSumProgram> &programs, const MlxFilter &filter) {
	std::vector<MlxSegment> segments;
	if (count > 0) {
		segments.push_back(SegmentFromHost(cols, count));
	}
	return EvalSegments(segments, programs, filter);
}

void MlxCacheDrop(const std::string &prefix) {
	std::lock_guard<std::mutex> guard(cache_mutex);
	auto &store = CacheStore();
	for (auto it = store.begin(); it != store.end();) {
		if (it->first.compare(0, prefix.size(), prefix) == 0) {
			it = store.erase(it);
		} else {
			++it;
		}
	}
}

bool MlxCacheHas(const std::vector<std::string> &keys, int64_t expected_rows) {
	std::lock_guard<std::mutex> guard(cache_mutex);
	auto &store = CacheStore();
	int64_t population = -1;
	for (auto &key : keys) {
		auto it = store.find(key);
		if (it == store.end() || it->second.rows != expected_rows) {
			return false;
		}
		if (population == -1) {
			population = it->second.population;
		} else if (population != it->second.population) {
			return false; // different populations are not row-aligned
		}
	}
	return true;
}

int64_t MlxCacheBeginPopulation(const std::string &table_prefix) {
	MlxCacheDrop(table_prefix);
	return ++cache_population_counter;
}

void MlxCacheStoreSegment(int64_t population, const std::vector<std::string> &col_keys,
                          const std::vector<MlxColumnData> &cols, size_t count) {
	if (count == 0) {
		return;
	}
	auto segment = SegmentFromHost(cols, count);
	std::lock_guard<std::mutex> guard(cache_mutex);
	auto &store = CacheStore();
	for (size_t i = 0; i < col_keys.size(); i++) {
		auto &cached = store[col_keys[i]];
		cached.segments.push_back(segment.cols[i]);
		cached.valids.push_back(segment.valids[i]);
		cached.rows += static_cast<int64_t>(count);
		cached.population = population;
	}
}

std::vector<MlxSumResult> MlxSumExprsCached(const std::vector<std::string> &col_keys,
                                            const std::vector<MlxSumProgram> &programs, const MlxFilter &filter) {
	std::vector<MlxSegment> segments;
	{
		std::lock_guard<std::mutex> guard(cache_mutex);
		auto &store = CacheStore();
		size_t nsegments = 0;
		for (auto &key : col_keys) {
			auto it = store.find(key);
			if (it == store.end()) {
				throw std::runtime_error("GPU cache miss for column '" + key + "'");
			}
			if (nsegments == 0) {
				nsegments = it->second.segments.size();
				segments.resize(nsegments);
				for (size_t s = 0; s < nsegments; s++) {
					segments[s].count = static_cast<int64_t>(it->second.segments[s].shape(0));
				}
			} else if (it->second.segments.size() != nsegments) {
				throw std::runtime_error("GPU cache segment mismatch for column '" + key + "'");
			}
			for (size_t s = 0; s < nsegments; s++) {
				segments[s].cols.push_back(it->second.segments[s]);
				segments[s].valids.push_back(it->second.valids[s]);
			}
		}
	}
	return EvalSegments(segments, programs, filter);
}

double MlxExprBenchInt64(const int64_t *data, size_t count) {
	if (count == 0) {
		return 0;
	}
	mx::array a(data, mx::Shape {static_cast<int>(count)}, mx::int64);
	auto f = mx::astype(a, mx::float32);
	auto expr = mx::add(mx::multiply(mx::sin(f), mx::cos(f)), mx::sqrt(mx::add(mx::abs(f), mx::array(1.0f))));
	return static_cast<double>(mx::sum(expr).item<float>());
}

int64_t MlxSumInt64(const int64_t *data, size_t count) {
	if (count == 0) {
		return 0;
	}
	mx::array a(data, mx::Shape {static_cast<int>(count)}, mx::int64);
	return mx::sum(a).item<int64_t>();
}

} // namespace duckdb_mlx
