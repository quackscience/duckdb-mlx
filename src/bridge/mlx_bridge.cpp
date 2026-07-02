#include "mlx_bridge.hpp"
#include "mlx_groupby_detail.hpp"

// This file must not include any DuckDB header (see mlx_bridge.hpp).
#include <mlx/mlx.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
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
		if (col.ivalues) {
			segment.cols.emplace_back(mx::array(col.ivalues, shape, mx::int64));
		} else {
			segment.cols.emplace_back(mx::array(col.values, shape, mx::float32));
		}
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
			if (op.int_lane) {
				stack.push_back(mx::array(op.ivalue, mx::int64));
			} else {
				stack.push_back(mx::array(static_cast<float>(op.value)));
			}
			break;
		case MlxExprOpCode::TO_FLOAT: {
			auto a = stack.back();
			stack.pop_back();
			stack.push_back(mx::multiply(mx::astype(a, mx::float32), mx::array(static_cast<float>(op.value))));
			break;
		}
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

//! Gather `data` rows where `mask` is true (host index build; compact then compute).
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
			// neutral elements per lane: the int lane reduces exactly in int64
			auto neutral_zero = program.int_lane ? mx::array(static_cast<int64_t>(0), mx::int64) : mx::array(0.0f);
			auto neutral_min = program.int_lane ? mx::array(std::numeric_limits<int64_t>::max(), mx::int64)
			                                    : mx::array(std::numeric_limits<float>::infinity());
			auto neutral_max = program.int_lane ? mx::array(std::numeric_limits<int64_t>::min(), mx::int64)
			                                    : mx::array(-std::numeric_limits<float>::infinity());
			switch (program.kind) {
			case MlxAggKind::MIN:
				outs.push_back(mask.has_value() ? mx::min(mx::where(*mask, expr, neutral_min)) : mx::min(expr));
				break;
			case MlxAggKind::MAX:
				outs.push_back(mask.has_value() ? mx::max(mx::where(*mask, expr, neutral_max)) : mx::max(expr));
				break;
			default: // SUM / AVG
				outs.push_back(mask.has_value() ? mx::sum(mx::where(*mask, expr, neutral_zero)) : mx::sum(expr));
				break;
			}
		}
		if (mask.has_value()) {
			// int32 reduce: segments are well under 2^31 rows and Metal has
			// no native 64-bit arithmetic
			outs.push_back(mx::sum(mx::astype(*mask, mx::int32)));
		}
	}
}

void MergeSegmentOutputs(const MlxSegment &segment, const std::vector<MlxSumProgram> &programs, const MlxFilter &filter,
                         const std::vector<mx::array> &outs, std::vector<MlxSumResult> &results) {
	size_t cursor = 0;
	for (size_t p = 0; p < programs.size(); p++) {
		auto &program = programs[p];
		double value = 0;
		int64_t ivalue = 0;
		if (HasValueGraph(program.kind)) {
			if (program.int_lane) {
				ivalue = outs[cursor++].item<int64_t>();
				value = static_cast<double>(ivalue);
			} else {
				value = static_cast<double>(outs[cursor++].item<float>());
			}
		}
		int64_t count = SegmentHasMask(segment, filter, program) ? static_cast<int64_t>(outs[cursor++].item<int32_t>())
		                                                         : segment.count;
		switch (program.kind) {
		case MlxAggKind::MIN:
			results[p].value = std::min(results[p].value, value);
			results[p].ivalue = std::min(results[p].ivalue, ivalue);
			break;
		case MlxAggKind::MAX:
			results[p].value = std::max(results[p].value, value);
			results[p].ivalue = std::max(results[p].ivalue, ivalue);
			break;
		case MlxAggKind::COUNT:
		case MlxAggKind::COUNT_STAR:
			break;
		default:
			results[p].value += value;
			results[p].ivalue += ivalue;
			break;
		}
		results[p].valid_count += count;
	}
}

std::vector<MlxSumResult> EvalSegments(const std::vector<MlxSegment> &segments,
                                       const std::vector<MlxSumProgram> &programs, const MlxFilter &filter,
                                       const std::vector<std::vector<MlxZoneMap>> *segment_zone_maps,
                                       MlxCacheStats *stats) {
	std::vector<MlxSumResult> results;
	for (auto &program : programs) {
		switch (program.kind) {
		case MlxAggKind::MIN:
			results.push_back({std::numeric_limits<double>::infinity(), 0, std::numeric_limits<int64_t>::max()});
			break;
		case MlxAggKind::MAX:
			results.push_back({-std::numeric_limits<double>::infinity(), 0, std::numeric_limits<int64_t>::min()});
			break;
		default:
			results.push_back({0.0, 0, 0});
			break;
		}
	}

	struct PrunePred {
		int32_t col;
		MlxExprOpCode code;
		double constant; // fp32-lane compare bound
		int64_t iconst;  // exact bound for int-lane columns
		bool int_lane;
	};
	std::vector<PrunePred> prune_preds;
	if (segment_zone_maps && !filter.ops.empty()) {
		size_t i = 0;
		auto parse_cmp = [&]() -> bool {
			if (i + 3 > filter.ops.size()) {
				return false;
			}
			if (filter.ops[i].code != MlxExprOpCode::LOAD_COL || filter.ops[i + 1].code != MlxExprOpCode::CONST_VAL) {
				return false;
			}
			auto cmp = filter.ops[i + 2].code;
			if (cmp < MlxExprOpCode::CMP_LT || cmp > MlxExprOpCode::CMP_NE) {
				return false;
			}
			auto &konst = filter.ops[i + 1];
			prune_preds.push_back({filter.ops[i].col, cmp, konst.value, konst.ivalue, konst.int_lane});
			i += 3;
			return true;
		};
		if (parse_cmp()) {
			while (i < filter.ops.size()) {
				if (filter.ops[i].code != MlxExprOpCode::AND) {
					prune_preds.clear();
					break;
				}
				i++;
				if (!parse_cmp()) {
					prune_preds.clear();
					break;
				}
			}
		}
	}

	auto segment_pruned = [&](size_t seg_idx) {
		if (!segment_zone_maps || prune_preds.empty()) {
			return false;
		}
		for (auto &pred : prune_preds) {
			if (pred.col < 0 || static_cast<size_t>(pred.col) >= segment_zone_maps->size()) {
				return false;
			}
			if (seg_idx >= (*segment_zone_maps)[pred.col].size()) {
				return false;
			}
			auto &zm = (*segment_zone_maps)[pred.col][seg_idx];
			if (pred.int_lane != zm.int_lane) {
				return false; // lane mismatch: never prune
			}
			// exact int64 bounds for the int lane; double bounds otherwise
			auto lo = zm.int_lane ? static_cast<double>(zm.imin) : zm.min_val;
			auto hi = zm.int_lane ? static_cast<double>(zm.imax) : zm.max_val;
			auto ilo = zm.imin;
			auto ihi = zm.imax;
			auto c = pred.constant;
			auto ic = pred.iconst;
			switch (pred.code) {
			case MlxExprOpCode::CMP_LT:
				if (zm.int_lane ? ilo >= ic : lo >= c) {
					return true;
				}
				break;
			case MlxExprOpCode::CMP_LE:
				if (zm.int_lane ? ilo > ic : lo > c) {
					return true;
				}
				break;
			case MlxExprOpCode::CMP_GT:
				if (zm.int_lane ? ihi <= ic : hi <= c) {
					return true;
				}
				break;
			case MlxExprOpCode::CMP_GE:
				if (zm.int_lane ? ihi < ic : hi < c) {
					return true;
				}
				break;
			case MlxExprOpCode::CMP_EQ:
				if (zm.int_lane ? (ic < ilo || ic > ihi) : (c < lo || c > hi)) {
					return true;
				}
				break;
			case MlxExprOpCode::CMP_NE:
				if (!zm.has_null && (zm.int_lane ? (ilo == ihi && ilo == ic) : (lo == hi && lo == c))) {
					return true;
				}
				break;
			default:
				return false;
			}
		}
		return false;
	};

	std::vector<mx::array> all;
	std::vector<const MlxSegment *> evaluated;
	size_t seg_idx = 0;
	for (auto &segment : segments) {
		if (stats) {
			stats->segments_total++;
		}
		if (segment.count == 0) {
			seg_idx++;
			continue;
		}
		if (segment_pruned(seg_idx)) {
			if (stats) {
				stats->segments_pruned++;
			}
			seg_idx++;
			continue;
		}
		// mx::compile fuses each segment's whole expression forest — masks
		// included — into a few kernels instead of materializing every
		// intermediate array. (A gather-then-compute path for highly selective
		// filters must be gated by zone-map selectivity, never by eagerly
		// evaluating the mask: that costs more than it saves.)
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
		seg_idx++;
	}
	mx::eval(all);

	size_t cursor = 0;
	for (auto segment : evaluated) {
		std::vector<mx::array> outs;
		for (size_t p = 0; p < programs.size(); p++) {
			auto &program = programs[p];
			if (HasValueGraph(program.kind)) {
				outs.push_back(all[cursor++]);
			}
			if (SegmentHasMask(*segment, filter, program)) {
				outs.push_back(all[cursor++]);
			}
		}
		MergeSegmentOutputs(*segment, programs, filter, outs, results);
	}
	for (size_t p = 0; p < programs.size(); p++) {
		if (programs[p].kind == MlxAggKind::AVG && results[p].valid_count > 0) {
			// int lane: exact int64 sum divided once (raw scale handled by
			// the planner when rendering)
			auto numerator = programs[p].int_lane ? static_cast<double>(results[p].ivalue) : results[p].value;
			results[p].value = numerator / static_cast<double>(results[p].valid_count);
		}
	}
	return results;
}

std::mutex cache_mutex;
std::atomic<int64_t> cache_population_counter {0};
std::atomic<int64_t> last_segments_total {0};
std::atomic<int64_t> last_segments_pruned {0};

struct MlxCachedColumn {
	std::vector<mx::array> segments;
	std::vector<std::optional<mx::array>> valids;
	std::vector<MlxZoneMap> zone_maps;
	int64_t rows = 0;
	int64_t population = 0;
};

struct MlxTableLayout {
	int64_t rows = 0;
	int64_t population = 0;
	std::vector<size_t> segment_sizes;
};

std::unordered_map<std::string, MlxCachedColumn> &CacheStore() {
	static std::unordered_map<std::string, MlxCachedColumn> store;
	return store;
}

std::unordered_map<std::string, MlxTableLayout> &TableLayouts() {
	static std::unordered_map<std::string, MlxTableLayout> layouts;
	return layouts;
}

template <class T>
MlxZoneMap ComputeZoneMapT(const T *values, const uint8_t *valid, size_t count) {
	MlxZoneMap zm;
	if (count == 0) {
		return zm;
	}
	T lo = std::numeric_limits<T>::max();
	T hi = std::numeric_limits<T>::lowest();
	bool any = false;
	for (size_t i = 0; i < count; i++) {
		if (valid && !valid[i]) {
			zm.has_null = true;
			continue;
		}
		lo = std::min(lo, values[i]);
		hi = std::max(hi, values[i]);
		any = true;
	}
	if (!any) {
		lo = 0;
		hi = 0;
	}
	zm.min_val = static_cast<double>(lo);
	zm.max_val = static_cast<double>(hi);
	if (std::is_integral<T>::value) {
		zm.int_lane = true;
		zm.imin = static_cast<int64_t>(lo);
		zm.imax = static_cast<int64_t>(hi);
	}
	return zm;
}

void DropTableUnlocked(const std::string &prefix) {
	auto &store = CacheStore();
	for (auto it = store.begin(); it != store.end();) {
		if (it->first.compare(0, prefix.size(), prefix) == 0) {
			it = store.erase(it);
		} else {
			++it;
		}
	}
	TableLayouts().erase(prefix);
	MlxGroupbyDenseClearTable(prefix);
}

void AppendColumnSlice(MlxCachedColumn &cached, const MlxColumnData &col, size_t offset, size_t take,
                       int64_t population) {
	if (take == 0) {
		return;
	}
	mx::Shape shape {static_cast<int>(take)};
	auto valid_ptr = col.valid ? col.valid + offset : nullptr;
	if (col.ivalues) {
		cached.zone_maps.push_back(ComputeZoneMapT<int64_t>(col.ivalues + offset, valid_ptr, take));
		cached.segments.push_back(mx::array(col.ivalues + offset, shape, mx::int64));
	} else {
		cached.zone_maps.push_back(ComputeZoneMapT<float>(col.values + offset, valid_ptr, take));
		cached.segments.push_back(mx::array(col.values + offset, shape, mx::float32));
	}
	if (col.valid) {
		cached.valids.push_back(mx::array(valid_ptr, shape, mx::uint8));
	} else {
		cached.valids.push_back(std::nullopt);
	}
	cached.rows += static_cast<int64_t>(take);
	cached.population = population;
}

std::vector<size_t> CanonicalSegmentSizesUnlocked(const std::string &table_prefix) {
	for (auto &entry : CacheStore()) {
		if (entry.first.compare(0, table_prefix.size(), table_prefix) == 0 && !entry.second.segments.empty()) {
			std::vector<size_t> sizes;
			for (auto &seg : entry.second.segments) {
				sizes.push_back(static_cast<size_t>(seg.shape(0)));
			}
			return sizes;
		}
	}
	auto &layout = TableLayouts()[table_prefix];
	return layout.segment_sizes;
}

} // namespace

std::vector<MlxSumResult> MlxSumExprs(const std::vector<MlxColumnData> &cols, size_t count,
                                      const std::vector<MlxSumProgram> &programs, const MlxFilter &filter) {
	std::vector<MlxSegment> segments;
	if (count > 0) {
		segments.push_back(SegmentFromHost(cols, count));
	}
	return EvalSegments(segments, programs, filter, nullptr, nullptr);
}

void MlxCacheDrop(const std::string &prefix) {
	std::lock_guard<std::mutex> guard(cache_mutex);
	DropTableUnlocked(prefix);
}

bool MlxCacheHas(const std::vector<std::string> &keys, int64_t expected_rows) {
	std::lock_guard<std::mutex> guard(cache_mutex);
	auto &store = CacheStore();
	size_t nsegments = 0;
	for (auto &key : keys) {
		auto it = store.find(key);
		if (it == store.end() || it->second.rows != expected_rows || it->second.segments.empty()) {
			return false;
		}
		if (nsegments == 0) {
			nsegments = it->second.segments.size();
		} else if (it->second.segments.size() != nsegments) {
			return false;
		}
	}
	return true;
}

MlxPopulationPlan MlxCacheBeginPopulation(const std::string &table_prefix, const std::vector<std::string> &col_keys,
                                          int64_t expected_rows) {
	std::lock_guard<std::mutex> guard(cache_mutex);
	MlxPopulationPlan plan;
	plan.store_col.assign(col_keys.size(), true);

	auto &store = CacheStore();
	auto &layouts = TableLayouts();

	// Row-count mismatch on any cached column in this table => full rebuild
	for (auto &entry : store) {
		if (entry.first.compare(0, table_prefix.size(), table_prefix) == 0 && entry.second.rows != expected_rows) {
			DropTableUnlocked(table_prefix);
			break;
		}
	}

	auto &layout = layouts[table_prefix];
	if (layout.rows == 0) {
		layout.rows = expected_rows;
		layout.population = ++cache_population_counter;
	} else if (layout.rows != expected_rows) {
		DropTableUnlocked(table_prefix);
		layout = {};
		layout.rows = expected_rows;
		layout.population = ++cache_population_counter;
	}
	plan.population = layout.population;

	bool any_present = false;
	bool any_missing = false;
	for (size_t i = 0; i < col_keys.size(); i++) {
		auto it = store.find(col_keys[i]);
		if (it != store.end() && it->second.rows == expected_rows && !it->second.segments.empty()) {
			plan.store_col[i] = false;
			any_present = true;
		} else {
			store.erase(col_keys[i]);
			any_missing = true;
		}
	}
	// partial miss within one query's column set => repopulate all keys together
	if (any_present && any_missing) {
		for (auto &key : col_keys) {
			store.erase(key);
		}
		plan.store_col.assign(col_keys.size(), true);
	}
	return plan;
}

void MlxCacheStoreSegment(int64_t population, const std::vector<std::string> &col_keys,
                          const std::vector<bool> &store_col, const std::vector<MlxColumnData> &cols, size_t count) {
	if (count == 0) {
		return;
	}
	std::string table_prefix;
	if (!col_keys.empty()) {
		auto pos = col_keys[0].rfind('#');
		if (pos != std::string::npos) {
			table_prefix = col_keys[0].substr(0, pos + 1);
		}
	}

	std::lock_guard<std::mutex> guard(cache_mutex);
	auto &store = CacheStore();
	auto &layout = TableLayouts()[table_prefix];
	auto canonical = CanonicalSegmentSizesUnlocked(table_prefix);

	if (canonical.empty()) {
		for (size_t i = 0; i < col_keys.size(); i++) {
			if (i >= store_col.size() || !store_col[i]) {
				continue;
			}
			AppendColumnSlice(store[col_keys[i]], cols[i], 0, count, population);
		}
		layout.segment_sizes.push_back(count);
		return;
	}

	// split incoming rows to match the table's canonical segment boundaries
	size_t offset = 0;
	size_t seg_idx = 0;
	for (size_t i = 0; i < col_keys.size(); i++) {
		if (i < store_col.size() && store_col[i] && !store[col_keys[i]].segments.empty()) {
			seg_idx = store[col_keys[i]].segments.size();
			break;
		}
	}
	while (offset < count) {
		size_t take = count - offset;
		if (seg_idx < canonical.size()) {
			take = std::min(take, canonical[seg_idx]);
		}
		for (size_t i = 0; i < col_keys.size(); i++) {
			if (i >= store_col.size() || !store_col[i]) {
				continue;
			}
			AppendColumnSlice(store[col_keys[i]], cols[i], offset, take, population);
		}
		offset += take;
		seg_idx++;
	}
}

std::vector<MlxSumResult> MlxSumExprsCached(const std::vector<std::string> &col_keys,
                                            const std::vector<MlxSumProgram> &programs, const MlxFilter &filter) {
	std::vector<MlxSegment> segments;
	std::vector<std::vector<MlxZoneMap>> segment_zone_maps;
	MlxCacheStats stats;
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
				segment_zone_maps.resize(col_keys.size());
				for (size_t s = 0; s < nsegments; s++) {
					segments[s].count = static_cast<int64_t>(it->second.segments[s].shape(0));
				}
			} else if (it->second.segments.size() != nsegments) {
				throw std::runtime_error("GPU cache segment mismatch for column '" + key + "'");
			}
		}
		for (size_t c = 0; c < col_keys.size(); c++) {
			auto &key = col_keys[c];
			auto it = store.find(key);
			segment_zone_maps[c] = it->second.zone_maps;
			for (size_t s = 0; s < nsegments; s++) {
				segments[s].cols.push_back(it->second.segments[s]);
				segments[s].valids.push_back(it->second.valids[s]);
			}
		}
	}
	auto results = EvalSegments(segments, programs, filter, &segment_zone_maps, &stats);
	last_segments_total.store(stats.segments_total);
	last_segments_pruned.store(stats.segments_pruned);
	return results;
}

MlxCacheStats MlxCacheLastStats() {
	MlxCacheStats stats;
	stats.segments_total = last_segments_total.load();
	stats.segments_pruned = last_segments_pruned.load();
	return stats;
}

std::vector<MlxZoneMap> MlxCacheColumnZoneMaps(const std::string &col_key) {
	std::lock_guard<std::mutex> guard(cache_mutex);
	auto it = CacheStore().find(col_key);
	if (it == CacheStore().end()) {
		return {};
	}
	return it->second.zone_maps;
}

void MlxCacheClearAll() {
	std::lock_guard<std::mutex> guard(cache_mutex);
	CacheStore().clear();
	TableLayouts().clear();
	MlxGroupbyDenseClearTable("");
}

void MlxGroupbyDenseAccumulateHost(const std::string &group_col_key, const std::string &value_col_key,
                                   int64_t population, const float *group_values, const float *sum_values,
                                   size_t count) {
	if (count == 0) {
		return;
	}
	mx::Shape shape {static_cast<int>(count)};
	auto keys = mx::astype(mx::round(mx::array(group_values, shape, mx::float32)), mx::int64);
	auto vals = mx::array(sum_values, shape, mx::float32);
	MlxGroupbyDenseAccumulate(group_col_key, value_col_key, population, keys, vals);
}

//! Whether a cached GROUP BY would be correct: the incremental dense table is
//! ready, or the cached fp32 columns are provably safe to re-derive exact
//! int64 keys from (integer keys within fp32-exact range, no NULLs anywhere).
bool MlxGroupbyCachedSafe(const std::string &group_col_key, const std::string &value_col_key) {
	int64_t population = -1;
	{
		std::lock_guard<std::mutex> guard(cache_mutex);
		auto &store = CacheStore();
		auto git = store.find(group_col_key);
		auto vit = store.find(value_col_key);
		if (git == store.end() || vit == store.end() || git->second.population != vit->second.population ||
		    git->second.segments.size() != vit->second.segments.size()) {
			return false;
		}
		population = git->second.population;
	}
	if (MlxGroupbyDenseReady(group_col_key, value_col_key, population)) {
		return true;
	}
	constexpr float kFp32Exact = 16777216.0f; // 2^24
	auto group_maps = MlxCacheColumnZoneMaps(group_col_key);
	if (group_maps.empty()) {
		return false;
	}
	for (auto &zm : group_maps) {
		if (zm.has_null || zm.min_val < -kFp32Exact || zm.max_val > kFp32Exact) {
			return false;
		}
	}
	for (auto &zm : MlxCacheColumnZoneMaps(value_col_key)) {
		if (zm.has_null) {
			return false;
		}
	}
	return true;
}

std::vector<MlxGroupbyRow> MlxGroupbySumCached(const std::string &group_col_key, const std::string &value_col_key) {
	int64_t population = -1;
	std::vector<MlxGroupbyRow> dense_rows;
	{
		std::lock_guard<std::mutex> guard(cache_mutex);
		auto &store = CacheStore();
		auto git = store.find(group_col_key);
		auto vit = store.find(value_col_key);
		if (git == store.end() || vit == store.end()) {
			throw std::runtime_error("GPU cache miss for GROUP BY columns");
		}
		if (git->second.population != vit->second.population ||
		    git->second.segments.size() != vit->second.segments.size()) {
			throw std::runtime_error("GPU cache segment mismatch for GROUP BY columns");
		}
		population = git->second.population;
	}
	if (MlxGroupbyDenseTryRead(group_col_key, value_col_key, population, dense_rows)) {
		return dense_rows;
	}

	// Fallback: concat GPU segments and scatter (e.g. cache populated without dense acc).
	std::vector<mx::array> key_segments;
	std::vector<mx::array> val_segments;
	{
		std::lock_guard<std::mutex> guard(cache_mutex);
		auto &store = CacheStore();
		key_segments = store[group_col_key].segments;
		val_segments = store[value_col_key].segments;
	}

	// Concatenate GPU-resident segments → one perfect-hash scatter (no per-segment sort).
	std::vector<mx::array> all_keys;
	std::vector<mx::array> all_vals;
	all_keys.reserve(key_segments.size());
	all_vals.reserve(val_segments.size());
	for (size_t s = 0; s < key_segments.size(); s++) {
		if (key_segments[s].shape(0) == 0) {
			continue;
		}
		all_keys.push_back(mx::astype(mx::round(key_segments[s]), mx::int64));
		all_vals.push_back(val_segments[s]);
	}
	if (all_keys.empty()) {
		return {};
	}
	mx::array keys = all_keys.size() == 1 ? all_keys[0] : mx::concatenate(all_keys, 0);
	mx::array vals = all_vals.size() == 1 ? all_vals[0] : mx::concatenate(all_vals, 0);
	// full path (dense window, else sort+scatter) — dense alone returns empty
	// for wide key spans
	return MlxGroupbySumArrays(keys, vals, false);
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
