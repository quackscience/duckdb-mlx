#include "mlx_bridge.hpp"
#include "mlx_groupby_detail.hpp"
#include "mlx_logger.hpp"

// This file must not include any DuckDB header (see mlx_bridge.hpp).
#include <mlx/mlx.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>
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

enum class EvalRowWidth : uint8_t { INT64, INT32, FP32 };

//! Evaluates a postfix program over the segment's columns.
mx::array EvalOps(const MlxSegment &segment, const std::vector<MlxExprOp> &ops,
                  EvalRowWidth width = EvalRowWidth::INT64) {
	std::vector<mx::array> stack;
	for (auto &op : ops) {
		switch (op.code) {
		case MlxExprOpCode::LOAD_COL: {
			auto col = segment.cols[op.col];
			if (width == EvalRowWidth::INT32) {
				stack.push_back(mx::astype(col, mx::int32));
			} else if (width == EvalRowWidth::FP32) {
				stack.push_back(mx::astype(col, mx::float32));
			} else {
				stack.push_back(col);
			}
			break;
		}
		case MlxExprOpCode::CONST_VAL:
			if (op.int_lane) {
				if (width == EvalRowWidth::INT32) {
					stack.push_back(mx::array(static_cast<int32_t>(op.ivalue), mx::int32));
				} else if (width == EvalRowWidth::FP32) {
					stack.push_back(mx::array(static_cast<float>(op.ivalue), mx::float32));
				} else {
					stack.push_back(mx::array(op.ivalue, mx::int64));
				}
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

namespace {

struct SegmentPrunePred {
	int32_t col;
	MlxExprOpCode code;
	double constant;
	int64_t iconst;
	bool int_lane;
};

bool BuildSegmentPrunePreds(const MlxFilter &filter, std::vector<SegmentPrunePred> &preds) {
	preds.clear();
	if (filter.ops.empty()) {
		return false;
	}
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
		preds.push_back({filter.ops[i].col, cmp, konst.value, konst.ivalue, konst.int_lane});
		i += 3;
		return true;
	};
	if (!parse_cmp()) {
		return false;
	}
	while (i < filter.ops.size()) {
		if (filter.ops[i].code != MlxExprOpCode::AND) {
			preds.clear();
			return false;
		}
		i++;
		if (!parse_cmp()) {
			preds.clear();
			return false;
		}
	}
	return !preds.empty();
}

bool SegmentShouldPrune(size_t seg_idx, const std::vector<SegmentPrunePred> &preds,
                        const std::vector<std::vector<MlxZoneMap>> &segment_zone_maps) {
	if (preds.empty()) {
		return false;
	}
	for (auto &pred : preds) {
		if (pred.col < 0 || static_cast<size_t>(pred.col) >= segment_zone_maps.size()) {
			return false;
		}
		if (seg_idx >= segment_zone_maps[pred.col].size()) {
			return false;
		}
		auto &zm = segment_zone_maps[pred.col][seg_idx];
		if (pred.int_lane != zm.int_lane) {
			return false;
		}
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
}

} // namespace

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

	std::vector<SegmentPrunePred> prune_preds;
	if (segment_zone_maps) {
		BuildSegmentPrunePreds(filter, prune_preds);
	}

	auto segment_pruned = [&](size_t seg_idx) {
		return segment_zone_maps && SegmentShouldPrune(seg_idx, prune_preds, *segment_zone_maps);
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
	//! One contiguous GPU-resident column (fused at pin / population end).
	std::optional<mx::array> fused_col;
	std::optional<mx::array> fused_valid;
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

std::unordered_map<std::string, std::unordered_map<uint64_t, std::string>> &DerivedManifest() {
	static std::unordered_map<std::string, std::unordered_map<uint64_t, std::string>> manifest;
	return manifest;
}

struct MlxQ1PinBundle {
	int32_t shipdate_storage = -1;
	int32_t group_card = 0;
	int32_t val_n = 0;
	std::vector<uint64_t> pack_fps;
};

std::unordered_map<std::string, MlxQ1PinBundle> &Q1PinBundles() {
	static std::unordered_map<std::string, MlxQ1PinBundle> bundles;
	return bundles;
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
	DerivedManifest().erase(prefix);
	Q1PinBundles().erase(prefix);
}

void AppendColumnSlice(MlxCachedColumn &cached, const MlxColumnData &col, size_t offset, size_t take,
                       int64_t population) {
	if (take == 0) {
		return;
	}
	cached.fused_col = std::nullopt;
	cached.fused_valid = std::nullopt;
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

mx::array ConcatArraysBalanced(const std::vector<mx::array> &parts) {
	if (parts.empty()) {
		return mx::zeros({0}, mx::float32);
	}
	if (parts.size() == 1) {
		return parts[0];
	}
	if (parts.size() == 2) {
		return mx::concatenate(parts, 0);
	}
	size_t mid = parts.size() / 2;
	return mx::concatenate({ConcatArraysBalanced({parts.begin(), parts.begin() + static_cast<std::ptrdiff_t>(mid)}),
	                        ConcatArraysBalanced({parts.begin() + static_cast<std::ptrdiff_t>(mid), parts.end()})},
	                       0);
}

void FuseCachedColumnUnlocked(MlxCachedColumn &cached) {
	if (cached.segments.empty() || cached.fused_col.has_value()) {
		return;
	}
	if (cached.segments.size() == 1) {
		cached.fused_col = cached.segments[0];
	} else {
		cached.fused_col = ConcatArraysBalanced(cached.segments);
	}
	std::vector<mx::array> valid_parts;
	for (auto &v : cached.valids) {
		if (v.has_value()) {
			valid_parts.push_back(*v);
		}
	}
	if (!valid_parts.empty()) {
		cached.fused_valid =
		    valid_parts.size() == 1 ? valid_parts[0] : ConcatArraysBalanced(valid_parts);
	}
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

int64_t StorageIndexFromCacheKey(const std::string &key) {
	auto pos = key.rfind('#');
	if (pos == std::string::npos) {
		return -1;
	}
	auto suffix = key.substr(pos + 1);
	if (suffix.compare(0, 2, "x:") == 0) {
		return -1;
	}
	try {
		return std::stoll(suffix);
	} catch (...) {
		return -1;
	}
}

uint64_t FingerprintOps(const std::vector<MlxExprOp> &ops, const std::vector<std::string> &col_keys) {
	uint64_t h = 14695981039346656037ULL;
	for (auto &op : ops) {
		h ^= static_cast<uint64_t>(op.code);
		h *= 1099511628211ULL;
		if (op.code == MlxExprOpCode::LOAD_COL) {
			auto storage = StorageIndexFromCacheKey(col_keys[static_cast<size_t>(op.col)]);
			h ^= static_cast<uint64_t>(storage + 1000000);
			h *= 1099511628211ULL;
		} else if (op.code == MlxExprOpCode::CONST_VAL) {
			if (op.int_lane) {
				h ^= static_cast<uint64_t>(op.ivalue);
			} else {
				uint64_t bits = 0;
				std::memcpy(&bits, &op.value, sizeof(bits));
				h ^= bits;
			}
			h *= 1099511628211ULL;
		}
	}
	return h;
}

MlxSegment CachedSegmentFromKeys(const std::vector<std::string> &col_keys) {
	auto &store = CacheStore();
	MlxSegment seg;
	for (auto &key : col_keys) {
		auto it = store.find(key);
		if (it == store.end() || !it->second.fused_col.has_value()) {
			throw std::runtime_error("MlxCacheMaterializeDerived: missing fused column '" + key + "'");
		}
		if (seg.count == 0) {
			seg.count = it->second.rows;
		} else if (it->second.rows != seg.count) {
			throw std::runtime_error("MlxCacheMaterializeDerived: row count mismatch for '" + key + "'");
		}
		seg.cols.push_back(*it->second.fused_col);
		seg.valids.push_back(it->second.fused_valid);
	}
	return seg;
}

std::optional<mx::array> CombineCachedValids(const std::vector<std::string> &col_keys) {
	std::vector<mx::array> parts;
	for (auto &key : col_keys) {
		auto it = CacheStore().find(key);
		if (it != CacheStore().end() && it->second.fused_valid.has_value()) {
			parts.push_back(*it->second.fused_valid);
		}
	}
	if (parts.empty()) {
		return std::nullopt;
	}
	if (parts.size() == 1) {
		return parts[0];
	}
	auto out = parts[0];
	for (size_t i = 1; i < parts.size(); i++) {
		out = mx::logical_and(out, mx::astype(parts[i], mx::bool_));
	}
	return mx::astype(out, mx::uint8);
}

void MaterializeDerivedUnlocked(const std::string &table_prefix, const std::string &derived_key,
                                const std::vector<std::string> &base_col_keys, const std::vector<MlxExprOp> &ops) {
	auto &store = CacheStore();
	if (store.find(derived_key) != store.end() && store[derived_key].fused_col.has_value()) {
		return;
	}
	auto seg = CachedSegmentFromKeys(base_col_keys);
	auto derived = EvalOps(seg, ops, EvalRowWidth::INT64);
	mx::eval({derived});
	MlxCachedColumn cached;
	cached.fused_col = derived;
	cached.rows = seg.count;
	cached.population = store[base_col_keys[0]].population;
	cached.fused_valid = CombineCachedValids(base_col_keys);
	store[derived_key] = std::move(cached);
	DerivedManifest()[table_prefix][FingerprintOps(ops, base_col_keys)] = derived_key;
}

} // namespace

void MlxCacheFuseTable(const std::string &table_prefix) {
	std::lock_guard<std::mutex> guard(cache_mutex);
	std::vector<mx::array> to_eval;
	for (auto &entry : CacheStore()) {
		if (entry.first.compare(0, table_prefix.size(), table_prefix) != 0) {
			continue;
		}
		FuseCachedColumnUnlocked(entry.second);
		if (entry.second.fused_col.has_value()) {
			to_eval.push_back(*entry.second.fused_col);
		}
		if (entry.second.fused_valid.has_value()) {
			to_eval.push_back(*entry.second.fused_valid);
		}
	}
	if (!to_eval.empty()) {
		mx::eval(to_eval);
	}
}

void MlxCacheMaterializeDerived(const std::string &table_prefix, const std::string &derived_suffix,
                                const std::vector<std::string> &base_col_keys, const std::vector<MlxExprOp> &ops) {
	std::lock_guard<std::mutex> guard(cache_mutex);
	auto derived_key = table_prefix + "x:" + derived_suffix;
	MaterializeDerivedUnlocked(table_prefix, derived_key, base_col_keys, ops);
}

void MlxCacheMaterializeLineitemTpch(const std::string &table_prefix, int col_extendedprice, int col_discount,
                                     int col_tax, int64_t decimal_one_scaled) {
	auto ep_key = table_prefix + std::to_string(col_extendedprice);
	auto disc_key = table_prefix + std::to_string(col_discount);
	auto tax_key = table_prefix + std::to_string(col_tax);
	std::vector<std::string> ep_disc_keys = {ep_key, disc_key};

	std::vector<MlxExprOp> disc_price_ops = {{MlxExprOpCode::LOAD_COL, 1, 0},
	                                         {MlxExprOpCode::CONST_VAL, 0, 0.0, decimal_one_scaled, true},
	                                         {MlxExprOpCode::SUB, 0, 0},
	                                         {MlxExprOpCode::LOAD_COL, 0, 0},
	                                         {MlxExprOpCode::MUL, 0, 0}};
	MlxCacheMaterializeDerived(table_prefix, "disc_price", ep_disc_keys, disc_price_ops);

	auto disc_price_key = table_prefix + "x:disc_price";
	std::vector<std::string> charge_keys = {disc_price_key, tax_key};
	std::vector<MlxExprOp> charge_ops = {{MlxExprOpCode::LOAD_COL, 1, 0},
	                                     {MlxExprOpCode::CONST_VAL, 0, 0.0, decimal_one_scaled, true},
	                                     {MlxExprOpCode::ADD, 0, 0},
	                                     {MlxExprOpCode::LOAD_COL, 0, 0},
	                                     {MlxExprOpCode::MUL, 0, 0}};
	MlxCacheMaterializeDerived(table_prefix, "charge", charge_keys, charge_ops);
}

namespace {

constexpr int64_t kQ1ShipdateCutoffDays = 10471; // DATE '1998-09-02'

uint64_t FingerprintLoadCol(const std::string &col_key) {
	std::vector<MlxExprOp> ops = {{MlxExprOpCode::LOAD_COL, 0, 0}};
	std::vector<std::string> keys = {col_key};
	return FingerprintOps(ops, keys);
}

void StoreFusedDerivedUnlocked(const std::string &key, mx::array arr, int64_t rows, int64_t population) {
	MlxCachedColumn cached;
	cached.fused_col = std::move(arr);
	cached.rows = rows;
	cached.population = population;
	CacheStore()[key] = std::move(cached);
}

bool FilterMatchesQ1Shipdate(const MlxFilter &filter, const std::vector<std::string> &col_keys, int32_t shipdate_storage) {
	auto col_is_shipdate = [&](int32_t col) -> bool {
		if (col == shipdate_storage) {
			return true;
		}
		if (col < 0 || static_cast<size_t>(col) >= col_keys.size()) {
			return false;
		}
		return StorageIndexFromCacheKey(col_keys[static_cast<size_t>(col)]) == shipdate_storage;
	};
	for (size_t i = 0; i + 2 < filter.ops.size(); i++) {
		if (filter.ops[i].code != MlxExprOpCode::LOAD_COL || filter.ops[i + 2].code != MlxExprOpCode::CMP_LE) {
			continue;
		}
		if (filter.ops[i + 1].code != MlxExprOpCode::CONST_VAL) {
			continue;
		}
		if (!col_is_shipdate(static_cast<int32_t>(filter.ops[i].col))) {
			continue;
		}
		auto &konst = filter.ops[i + 1];
		if (konst.int_lane) {
			return konst.ivalue == kQ1ShipdateCutoffDays;
		}
		return konst.value == static_cast<double>(kQ1ShipdateCutoffDays);
	}
	return false;
}

void MaterializeLineitemQ1Unlocked(const std::string &table_prefix, int col_shipdate, int col_returnflag,
                                   int col_linestatus, int col_quantity, int col_extendedprice, int col_discount) {
	auto &store = CacheStore();
	auto ship_key = table_prefix + std::to_string(col_shipdate);
	auto rf_key = table_prefix + std::to_string(col_returnflag);
	auto ls_key = table_prefix + std::to_string(col_linestatus);
	auto qty_key = table_prefix + std::to_string(col_quantity);
	auto ep_key = table_prefix + std::to_string(col_extendedprice);
	auto disc_key = table_prefix + std::to_string(col_discount);
	auto disc_price_key = table_prefix + "x:disc_price";
	auto charge_key = table_prefix + "x:charge";
	for (auto &key : {ship_key, rf_key, ls_key, qty_key, ep_key, disc_key, disc_price_key, charge_key}) {
		auto it = store.find(key);
		if (it == store.end() || !it->second.fused_col.has_value()) {
			return;
		}
	}
	auto n = store[ship_key].rows;
	auto population = store[ship_key].population;

	auto &ship = *store[ship_key].fused_col;
	mx::array pass = ship.dtype() == mx::float32
	                     ? mx::astype(mx::less_equal(ship, mx::array(static_cast<float>(kQ1ShipdateCutoffDays))),
	                                  mx::uint8)
	                     : mx::astype(mx::less_equal(ship, mx::array(kQ1ShipdateCutoffDays, mx::int64)), mx::uint8);

	auto rf_i = mx::astype(mx::round(*store[rf_key].fused_col), mx::int32);
	auto ls_i = mx::astype(mx::round(*store[ls_key].fused_col), mx::int32);
	auto ls_card = mx::add(mx::max(ls_i), mx::array(1, mx::int32)).item<int32_t>();
	auto codes = mx::add(mx::multiply(rf_i, mx::array(ls_card, mx::int32)), ls_i);

	std::vector<mx::array> pack_cols = {*store[qty_key].fused_col, *store[ep_key].fused_col,
	                                    *store[disc_price_key].fused_col, *store[charge_key].fused_col,
	                                    *store[qty_key].fused_col, *store[ep_key].fused_col, *store[disc_key].fused_col};
	std::vector<mx::array> pack_up;
	pack_up.reserve(pack_cols.size());
	for (auto &col : pack_cols) {
		auto lane = col.dtype() == mx::int64 ? col : mx::astype(col, mx::int64);
		pack_up.push_back(mx::expand_dims(lane, 1));
	}
	auto packed = mx::concatenate(pack_up, 1);

	mx::eval({pass, codes, packed});

	StoreFusedDerivedUnlocked(table_prefix + "x:pass_q1", pass, n, population);
	StoreFusedDerivedUnlocked(table_prefix + "x:q1_codes", codes, n, population);
	StoreFusedDerivedUnlocked(table_prefix + "x:q1_pack", packed, n, population);

	MlxQ1PinBundle bundle;
	bundle.shipdate_storage = col_shipdate;
	bundle.group_card = mx::add(mx::max(codes), mx::array(1, mx::int32)).item<int32_t>();
	bundle.val_n = static_cast<int32_t>(pack_cols.size());
	bundle.pack_fps = {FingerprintLoadCol(qty_key),     FingerprintLoadCol(ep_key), FingerprintLoadCol(disc_price_key),
	                   FingerprintLoadCol(charge_key), FingerprintLoadCol(qty_key), FingerprintLoadCol(ep_key),
	                   FingerprintLoadCol(disc_key)};
	Q1PinBundles()[table_prefix] = std::move(bundle);
}

bool TryLineitemQ1FastPath(MlxGroupedState &state, const std::string &table_prefix,
                           const std::vector<std::string> &col_keys, const std::vector<MlxSumProgram> &programs,
                           const MlxFilter &filter) {
	auto bundle_it = Q1PinBundles().find(table_prefix);
	if (bundle_it == Q1PinBundles().end()) {
		duckdb_mlx::LogDebug("MLX Q1 fast path: no pin bundle");
		return false;
	}
	auto &bundle = bundle_it->second;
	if (!FilterMatchesQ1Shipdate(filter, col_keys, bundle.shipdate_storage)) {
		duckdb_mlx::LogDebug("MLX Q1 fast path: filter mismatch (ops=" + std::to_string(filter.ops.size()) + ")");
		return false;
	}
	if (state.card != bundle.group_card || bundle.val_n <= 0) {
		duckdb_mlx::LogDebug("MLX Q1 fast path: card mismatch state=" + std::to_string(state.card) + " bundle=" +
		                       std::to_string(bundle.group_card));
		return false;
	}
	std::vector<std::pair<size_t, int>> val_slots;
	val_slots.reserve(static_cast<size_t>(bundle.val_n));
	int pack_col = 0;
	for (size_t p = 0; p < programs.size(); p++) {
		auto kind = programs[p].kind;
		if (kind != MlxAggKind::SUM && kind != MlxAggKind::AVG) {
			continue;
		}
		if (!programs[p].int_lane || !HasValueGraph(kind)) {
			return false;
		}
		if (pack_col >= bundle.val_n) {
			return false;
		}
		val_slots.push_back({p, pack_col++});
	}
	// TPC-H Q1: 7 SUM/AVG lanes in pack column order (qty, ep, disc_price, charge, qty, ep, disc).
	if (pack_col != bundle.val_n || val_slots.empty()) {
		duckdb_mlx::LogDebug("MLX Q1 fast path: pack layout mismatch pack_col=" + std::to_string(pack_col) +
		                     " val_n=" + std::to_string(bundle.val_n));
		return false;
	}

	std::optional<mx::array> codes;
	std::optional<mx::array> pass;
	std::optional<mx::array> packed;
	{
		std::lock_guard<std::mutex> guard(cache_mutex);
		auto &store = CacheStore();
		auto codes_it = store.find(table_prefix + "x:q1_codes");
		auto pass_it = store.find(table_prefix + "x:pass_q1");
		auto pack_it = store.find(table_prefix + "x:q1_pack");
		if (codes_it == store.end() || pass_it == store.end() || pack_it == store.end() ||
		    !codes_it->second.fused_col.has_value() || !pass_it->second.fused_col.has_value() ||
		    !pack_it->second.fused_col.has_value()) {
			return false;
		}
		codes = *codes_it->second.fused_col;
		pass = *pass_it->second.fused_col;
		packed = *pack_it->second.fused_col;
	}
	if (!GroupedTileKernelEligible(programs, state.card)) {
		return false;
	}
	try {
		GroupedQ1FusedAccumulate(state, *codes, *pass, *packed, static_cast<int>(val_slots.size()), val_slots, programs);
		duckdb_mlx::LogDebug("MLX Q1 fast path: fused grid-stride accumulate (" +
		                     std::to_string(codes->shape(0)) + " rows)");
	} catch (...) {
		GroupedPackedTileAccumulate(state, *codes, *pass, *packed, static_cast<int>(val_slots.size()), val_slots, programs);
		duckdb_mlx::LogDebug("MLX Q1 fast path: fused failed, packed tile fallback");
	}
	return true;
}

} // namespace

void MlxCacheMaterializeLineitemQ1(const std::string &table_prefix, int col_shipdate, int col_returnflag,
                                   int col_linestatus, int col_quantity, int col_extendedprice, int col_discount) {
	std::lock_guard<std::mutex> guard(cache_mutex);
	MaterializeLineitemQ1Unlocked(table_prefix, col_shipdate, col_returnflag, col_linestatus, col_quantity,
	                              col_extendedprice, col_discount);
}

size_t MlxCacheBindDerivedPrograms(const std::string &table_prefix, std::vector<std::string> &col_keys,
                                   std::vector<MlxSumProgram> &programs) {
	std::lock_guard<std::mutex> guard(cache_mutex);
	auto manifest_it = DerivedManifest().find(table_prefix);
	if (manifest_it == DerivedManifest().end()) {
		return 0;
	}
	auto &manifest = manifest_it->second;
	auto &store = CacheStore();
	size_t bound = 0;
	for (auto &program : programs) {
		if (!HasValueGraph(program.kind)) {
			continue;
		}
		auto fp = FingerprintOps(program.ops, col_keys);
		auto derived_it = manifest.find(fp);
		if (derived_it == manifest.end()) {
			continue;
		}
		auto &derived_key = derived_it->second;
		auto cache_it = store.find(derived_key);
		if (cache_it == store.end() || !cache_it->second.fused_col.has_value()) {
			continue;
		}
		int32_t col_idx = -1;
		for (size_t i = 0; i < col_keys.size(); i++) {
			if (col_keys[i] == derived_key) {
				col_idx = static_cast<int32_t>(i);
				break;
			}
		}
		if (col_idx < 0) {
			col_keys.push_back(derived_key);
			col_idx = static_cast<int32_t>(col_keys.size() - 1);
		}
		program.ops = {{MlxExprOpCode::LOAD_COL, col_idx, 0}};
		bound++;
	}
	return bound;
}

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
	DerivedManifest().clear();
	Q1PinBundles().clear();
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

//===--------------------------------------------------------------------===//
// Generalized grouped aggregation: GPU evaluates codes/masks/expressions,
// then scatter_add into dense accumulators (GQE cuDF-style groupby on Metal).
//===--------------------------------------------------------------------===//

namespace {

struct GroupedSegmentArrays {
	mx::array codes;                              // int32; == card for filtered rows
	std::vector<std::optional<mx::array>> exprs;  // per program value arrays
	std::vector<std::optional<mx::array>> masks;  // per program NULL masks (uint8)
	int64_t count = 0;

	GroupedSegmentArrays() : codes(mx::zeros({0}, mx::int32)) {
	}
};

EvalRowWidth ProgramEvalWidth(const MlxSumProgram &program) {
	if (program.int_lane && program.int32_rows) {
		return EvalRowWidth::INT32;
	}
	if (program.int_lane && program.gpu_fp32_rows) {
		return EvalRowWidth::FP32;
	}
	return EvalRowWidth::INT64;
}

int64_t GpuScatterChunkRows(const std::vector<MlxSumProgram> &programs) {
	int64_t chunk = 2048;
	for (auto &program : programs) {
		if (!program.int_lane || (program.kind != MlxAggKind::SUM && program.kind != MlxAggKind::AVG)) {
			continue;
		}
		if (!program.int32_rows && !program.gpu_fp32_rows) {
			continue;
		}
		auto bound = std::max(program.row_abs_bound, 1.0);
		auto limit = program.int32_rows ? (2147483647.0 / bound) : (16777216.0 / bound);
		chunk = std::min(chunk, std::max<int64_t>(1, static_cast<int64_t>(limit)));
	}
	return chunk;
}

GroupedSegmentArrays SliceGroupedSegment(const GroupedSegmentArrays &seg, int64_t begin, int64_t end) {
	GroupedSegmentArrays out;
	out.count = end - begin;
	out.codes = mx::slice(seg.codes, {static_cast<int>(begin)}, {static_cast<int>(end)});
	for (auto &e : seg.exprs) {
		out.exprs.push_back(e.has_value()
		                        ? std::optional<mx::array>(
		                              mx::slice(*e, {static_cast<int>(begin)}, {static_cast<int>(end)}))
		                        : std::nullopt);
	}
	for (auto &m : seg.masks) {
		out.masks.push_back(m.has_value()
		                        ? std::optional<mx::array>(
		                              mx::slice(*m, {static_cast<int>(begin)}, {static_cast<int>(end)}))
		                        : std::nullopt);
	}
	return out;
}

struct GroupedGpuAcc {
	mx::array rows;   // (card,) int32 — filter-passing row counts
	mx::array counts; // (card * nprogs,) int32
	mx::array isums;  // (card * nprogs,) int32 — int-lane SUM/AVG partials
	mx::array fsums;  // (card * nprogs,) float32
	int64_t card = 0;
	size_t nprogs = 0;

	GroupedGpuAcc(int64_t card_p, size_t nprogs_p)
	    : rows(mx::zeros({static_cast<int>(card_p)}, mx::int32)),
	      counts(mx::zeros({static_cast<int>(card_p * static_cast<int64_t>(nprogs_p))}, mx::int32)),
	      isums(mx::zeros({static_cast<int>(card_p * static_cast<int64_t>(nprogs_p))}, mx::int32)),
	      fsums(mx::zeros({static_cast<int>(card_p * static_cast<int64_t>(nprogs_p))}, mx::float32)), card(card_p),
	      nprogs(nprogs_p) {
	}
};

std::mutex &GroupedGpuMutex() {
	static std::mutex m;
	return m;
}

std::unordered_map<int64_t, GroupedGpuAcc> &GroupedGpuStore() {
	static std::unordered_map<int64_t, GroupedGpuAcc> store;
	return store;
}

std::atomic<int64_t> &GroupedGpuNextId() {
	static std::atomic<int64_t> next_id {0};
	return next_id;
}

bool ProgramsNeedCpuScatter(const std::vector<MlxSumProgram> &programs) {
	(void)programs;
	return false;
}

int64_t GroupedGpuOpen(const MlxGroupedState &state, const std::vector<MlxSumProgram> &programs) {
	if (state.card <= 0 || state.card > 65536) {
		return -1;
	}
	GroupedGpuAcc acc(state.card, programs.size());
	auto id = GroupedGpuNextId().fetch_add(1);
	std::lock_guard<std::mutex> guard(GroupedGpuMutex());
	GroupedGpuStore().emplace(id, std::move(acc));
	return id;
}

void GroupedGpuClose(int64_t handle) {
	std::lock_guard<std::mutex> guard(GroupedGpuMutex());
	GroupedGpuStore().erase(handle);
}

GroupedGpuAcc *GroupedGpuLookup(int64_t handle) {
	std::lock_guard<std::mutex> guard(GroupedGpuMutex());
	auto it = GroupedGpuStore().find(handle);
	return it == GroupedGpuStore().end() ? nullptr : &it->second;
}

static constexpr int64_t kDenseReduceCard = 512;
//! Max rows per fused cached eval for low-cardinality GROUP BY (few syncs).
static constexpr int64_t kDenseCachedBatchRows = 8 * 1024 * 1024;

mx::array ConcatArrays(const std::vector<mx::array> &parts) {
	if (parts.empty()) {
		return mx::zeros({0}, mx::float32);
	}
	if (parts.size() == 1) {
		return parts[0];
	}
	if (parts.size() == 2) {
		return mx::concatenate(parts, 0);
	}
	size_t mid = parts.size() / 2;
	return mx::concatenate({ConcatArrays({parts.begin(), parts.begin() + static_cast<std::ptrdiff_t>(mid)}),
	                        ConcatArrays({parts.begin() + static_cast<std::ptrdiff_t>(mid), parts.end()})},
	                       0);
}

MlxSegment ConcatSegmentParts(const std::vector<MlxSegment> &segments, size_t begin, size_t end) {
	MlxSegment out;
	if (begin >= end) {
		return out;
	}
	auto ncols = segments[begin].cols.size();
	out.cols.reserve(ncols);
	out.valids.reserve(ncols);
	for (size_t c = 0; c < ncols; c++) {
		std::vector<mx::array> col_parts;
		std::vector<mx::array> val_parts;
		bool has_valid = false;
		for (size_t s = begin; s < end; s++) {
			if (segments[s].count <= 0) {
				continue;
			}
			col_parts.push_back(segments[s].cols[c]);
			if (segments[s].valids[c].has_value()) {
				has_valid = true;
				val_parts.push_back(*segments[s].valids[c]);
			}
		}
		if (col_parts.empty()) {
			out.cols.push_back(mx::zeros({0}, segments[begin].cols[c].dtype()));
		} else {
			out.cols.push_back(ConcatArrays(col_parts));
		}
		if (has_valid) {
			out.valids.push_back(ConcatArrays(val_parts));
		} else {
			out.valids.push_back(std::nullopt);
		}
	}
	out.count = 0;
	for (size_t s = begin; s < end; s++) {
		out.count += segments[s].count;
	}
	return out;
}

void BuildGroupedEvalGraphs(const MlxSegment &segment, const MlxGroupedSpec &spec,
                            const std::vector<MlxSumProgram> &programs, const MlxFilter &filter, bool exact_int_eval,
                            std::vector<mx::array> &outs) {
	int64_t card = 1;
	for (auto c : spec.key_cards) {
		card *= c;
	}
	std::optional<mx::array> code;
	for (size_t k = 0; k < spec.key_cols.size(); k++) {
		auto &col = segment.cols[spec.key_cols[k]];
		mx::array c32 = col.dtype() == mx::int64
		                    ? mx::astype(mx::subtract(col, mx::array(spec.key_offsets[k], mx::int64)), mx::int32)
		                    : mx::astype(mx::subtract(col, mx::array(static_cast<float>(spec.key_offsets[k]))),
		                                 mx::int32);
		code = code.has_value()
		           ? mx::add(mx::multiply(*code, mx::array(static_cast<int32_t>(spec.key_cards[k]), mx::int32)), c32)
		           : c32;
	}
	std::optional<mx::array> fmask;
	if (!filter.ops.empty()) {
		fmask = EvalOps(segment, filter.ops);
	}
	fmask = CombineMasks(fmask, NullMask(segment, filter.null_cols));
	auto dump = mx::array(static_cast<int32_t>(card), mx::int32);
	auto in_range = mx::logical_and(mx::greater_equal(*code, mx::array(0, mx::int32)), mx::less(*code, dump));
	auto guarded = fmask.has_value() ? mx::logical_and(in_range, *fmask) : in_range;
	outs.push_back(mx::where(guarded, *code, dump));

	for (auto &program : programs) {
		if (HasValueGraph(program.kind)) {
			auto width = ProgramEvalWidth(program);
			if (exact_int_eval && program.int_lane) {
				width = EvalRowWidth::INT64;
			}
			outs.push_back(EvalOps(segment, program.ops, width));
		}
	}
	for (auto &program : programs) {
		auto nmask = NullMask(segment, program.null_cols);
		if (nmask.has_value()) {
			outs.push_back(mx::astype(*nmask, mx::uint8));
		}
	}
}

GroupedSegmentArrays GroupedEvalSegment(const MlxSegment &segment, const MlxGroupedSpec &spec,
                                        const std::vector<MlxSumProgram> &programs, const MlxFilter &filter,
                                        bool exact_int_eval = false, bool defer_materialize = false) {
	GroupedSegmentArrays out;
	out.count = segment.count;
	if (segment.count == 0) {
		out.exprs.resize(programs.size());
		out.masks.resize(programs.size());
		return out;
	}

	std::vector<mx::array> inputs = segment.cols;
	std::vector<int32_t> valid_slots(segment.valids.size(), -1);
	for (size_t i = 0; i < segment.valids.size(); i++) {
		if (segment.valids[i].has_value()) {
			valid_slots[i] = static_cast<int32_t>(inputs.size());
			inputs.push_back(*segment.valids[i]);
		}
	}
	std::vector<mx::array> graphs;
	if (defer_materialize) {
		// Resident hot path: skip mx::compile JIT — caller fuses eval with tile reduce.
		BuildGroupedEvalGraphs(segment, spec, programs, filter, exact_int_eval, graphs);
	} else {
		auto ncols = segment.cols.size();
		auto count = segment.count;
		auto fn = [spec, programs, filter, exact_int_eval, ncols, valid_slots, count](const std::vector<mx::array> &ins) {
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
			BuildGroupedEvalGraphs(seg, spec, programs, filter, exact_int_eval, outs);
			return outs;
		};
		graphs = mx::compile(fn)(inputs);
	}
	size_t cursor = 0;
	out.codes = graphs[cursor++];
	out.exprs.resize(programs.size());
	for (size_t p = 0; p < programs.size(); p++) {
		if (HasValueGraph(programs[p].kind)) {
			out.exprs[p] = graphs[cursor++];
		} else {
			out.exprs[p] = std::nullopt;
		}
	}
	out.masks.resize(programs.size());
	for (size_t p = 0; p < programs.size(); p++) {
		auto nmask = NullMask(segment, programs[p].null_cols);
		if (nmask.has_value()) {
			out.masks[p] = graphs[cursor++];
		} else {
			out.masks[p] = std::nullopt;
		}
	}

	std::vector<mx::array> to_eval = {out.codes};
	for (auto &e : out.exprs) {
		if (e.has_value()) {
			to_eval.push_back(*e);
		}
	}
	for (auto &m : out.masks) {
		if (m.has_value()) {
			to_eval.push_back(*m);
		}
	}
	if (!defer_materialize) {
		mx::eval(to_eval);
	}
	return out;
}

void GroupedCpuScatterInt(MlxGroupedState &state, const GroupedSegmentArrays &seg, const std::vector<MlxSumProgram> &programs) {
	auto n = seg.count;
	if (n == 0) {
		return;
	}
	auto card = state.card;
	auto nprogs = programs.size();
	auto codes = seg.codes.data<int32_t>();
	for (size_t p = 0; p < nprogs; p++) {
		if (!programs[p].int_lane || !seg.exprs[p].has_value()) {
			continue;
		}
		auto kind = programs[p].kind;
		if (kind == MlxAggKind::SUM || kind == MlxAggKind::AVG) {
			if (programs[p].int32_rows || programs[p].gpu_fp32_rows) {
				continue;
			}
		} else if (kind != MlxAggKind::MIN && kind != MlxAggKind::MAX) {
			continue;
		}
		const int64_t *iexpr = seg.exprs[p]->data<int64_t>();
		const int32_t *i32expr = programs[p].int32_rows ? seg.exprs[p]->data<int32_t>() : nullptr;
		const uint8_t *pmask = seg.masks[p].has_value() ? seg.masks[p]->data<uint8_t>() : nullptr;
		for (size_t i = 0; i < static_cast<size_t>(n); i++) {
			auto g = codes[i];
			if (g < 0 || g >= card || (pmask && !pmask[i])) {
				continue;
			}
			auto slot = static_cast<size_t>(g) * nprogs + p;
			auto v = i32expr ? static_cast<__int128>(i32expr[i]) : static_cast<__int128>(iexpr[i]);
			if (kind == MlxAggKind::SUM || kind == MlxAggKind::AVG) {
				state.ivalues[slot] += v;
			} else if (kind == MlxAggKind::MIN) {
				state.ivalues[slot] = std::min(state.ivalues[slot], v);
			} else {
				state.ivalues[slot] = std::max(state.ivalues[slot], v);
			}
		}
	}
}

void GroupedDenseGpuReduceMerge(MlxGroupedState &state, const GroupedSegmentArrays &seg,
                                const std::vector<MlxSumProgram> &programs, std::vector<mx::array> &outs) {
	auto card = state.card;
	auto nprogs = programs.size();
	size_t cursor = 0;
	for (int64_t g = 0; g < card; g++) {
		state.rows[static_cast<size_t>(g)] += outs[cursor++].item<int32_t>();
	}
	for (size_t p = 0; p < nprogs; p++) {
		auto kind = programs[p].kind;
		if (kind == MlxAggKind::COUNT_STAR) {
			continue;
		}
		for (int64_t g = 0; g < card; g++) {
			auto slot = static_cast<size_t>(g) * nprogs + p;
			if (kind == MlxAggKind::COUNT || !seg.exprs[p].has_value()) {
				state.counts[slot] += outs[cursor++].item<int32_t>();
				continue;
			}
			switch (kind) {
			case MlxAggKind::SUM:
			case MlxAggKind::AVG:
				if (programs[p].int_lane) {
					state.ivalues[slot] += static_cast<__int128>(outs[cursor++].item<int64_t>());
				} else {
					state.fvalues[slot] += static_cast<double>(outs[cursor++].item<float>());
				}
				state.counts[slot] += outs[cursor++].item<int32_t>();
				break;
			case MlxAggKind::MIN:
			case MlxAggKind::MAX: {
				auto cnt = outs[cursor + 1].item<int32_t>();
				if (cnt > 0) {
					if (programs[p].int_lane) {
						auto v = static_cast<__int128>(outs[cursor].item<int64_t>());
						state.ivalues[slot] = kind == MlxAggKind::MIN ? std::min(state.ivalues[slot], v)
						                                              : std::max(state.ivalues[slot], v);
					} else {
						auto v = static_cast<double>(outs[cursor].item<float>());
						state.fvalues[slot] = kind == MlxAggKind::MIN ? std::min(state.fvalues[slot], v)
						                                              : std::max(state.fvalues[slot], v);
					}
				}
				cursor += 2;
				break;
			}
			default:
				break;
			}
		}
	}
}

void GroupedDenseGpuReduce(MlxGroupedState &state, const GroupedSegmentArrays &seg,
                           const std::vector<MlxSumProgram> &programs, bool defer_materialize = false,
                           std::vector<mx::array> *reduce_outs = nullptr) {
	if (seg.count == 0) {
		return;
	}
	auto card = state.card;
	auto nprogs = programs.size();
	auto dump = mx::array(static_cast<int32_t>(card), mx::int32);
	auto zero_i32 = mx::array(static_cast<int32_t>(0), mx::int32);
	auto in_range = mx::logical_and(mx::greater_equal(seg.codes, zero_i32), mx::less(seg.codes, dump));

	std::vector<mx::array> gmasks;
	gmasks.reserve(static_cast<size_t>(card));
	for (int64_t g = 0; g < card; g++) {
		auto gcode = mx::array(static_cast<int32_t>(g), mx::int32);
		gmasks.push_back(mx::logical_and(in_range, mx::equal(seg.codes, gcode)));
	}

	std::vector<mx::array> outs;
	for (int64_t g = 0; g < card; g++) {
		outs.push_back(mx::sum(mx::astype(gmasks[static_cast<size_t>(g)], mx::int32)));
	}

	for (size_t p = 0; p < nprogs; p++) {
		auto kind = programs[p].kind;
		if (kind == MlxAggKind::COUNT_STAR) {
			continue;
		}
		for (int64_t g = 0; g < card; g++) {
			auto gmask = gmasks[static_cast<size_t>(g)];
			if (seg.masks[p].has_value()) {
				gmask = mx::logical_and(gmask, *seg.masks[p]);
			}
			if (kind == MlxAggKind::COUNT || !seg.exprs[p].has_value()) {
				outs.push_back(mx::sum(mx::astype(gmask, mx::int32)));
				continue;
			}
			auto val = *seg.exprs[p];
			switch (kind) {
			case MlxAggKind::SUM:
			case MlxAggKind::AVG: {
				auto neutral = programs[p].int_lane ? mx::array(static_cast<int64_t>(0), mx::int64) : mx::array(0.0f);
				outs.push_back(mx::sum(mx::where(gmask, val, neutral)));
				outs.push_back(mx::sum(mx::astype(gmask, mx::int32)));
				break;
			}
			case MlxAggKind::MIN: {
				auto neutral = programs[p].int_lane
				                   ? mx::array(std::numeric_limits<int64_t>::max(), mx::int64)
				                   : mx::array(std::numeric_limits<float>::infinity());
				outs.push_back(mx::min(mx::where(gmask, val, neutral)));
				outs.push_back(mx::sum(mx::astype(gmask, mx::int32)));
				break;
			}
			case MlxAggKind::MAX: {
				auto neutral = programs[p].int_lane
				                   ? mx::array(std::numeric_limits<int64_t>::min(), mx::int64)
				                   : mx::array(-std::numeric_limits<float>::infinity());
				outs.push_back(mx::max(mx::where(gmask, val, neutral)));
				outs.push_back(mx::sum(mx::astype(gmask, mx::int32)));
				break;
			}
			default:
				break;
			}
		}
	}
	if (defer_materialize) {
		if (reduce_outs) {
			*reduce_outs = std::move(outs);
		}
		return;
	}
	mx::eval(outs);
	GroupedDenseGpuReduceMerge(state, seg, programs, outs);
}

void GroupedGpuFlushPartials(MlxGroupedState &state, GroupedGpuAcc &acc, const std::vector<MlxSumProgram> &programs) {
	mx::eval({acc.isums, acc.fsums});
	auto isums_ptr = acc.isums.data<int32_t>();
	auto fsums_ptr = acc.fsums.data<float>();
	bool used_isums = false;
	bool used_fp32 = false;
	for (int64_t g = 0; g < acc.card; g++) {
		for (size_t p = 0; p < acc.nprogs; p++) {
			auto kind = programs[p].kind;
			if (!programs[p].int_lane || (kind != MlxAggKind::SUM && kind != MlxAggKind::AVG)) {
				continue;
			}
			auto slot = static_cast<size_t>(g) * acc.nprogs + p;
			if (programs[p].int32_rows) {
				state.ivalues[slot] += static_cast<__int128>(isums_ptr[slot]);
				used_isums = true;
			} else if (programs[p].gpu_fp32_rows) {
				state.ivalues[slot] += static_cast<__int128>(std::llround(static_cast<double>(fsums_ptr[slot])));
				used_fp32 = true;
			}
		}
	}
	if (used_isums) {
		acc.isums = mx::zeros_like(acc.isums);
	}
	if (used_fp32) {
		acc.fsums = mx::zeros_like(acc.fsums);
	}
}

void GroupedGpuScatter(GroupedGpuAcc &acc, const GroupedSegmentArrays &seg, const std::vector<MlxSumProgram> &programs) {
	if (seg.count == 0) {
		return;
	}
	auto nprogs = acc.nprogs;
	auto codes = seg.codes;
	auto dump = mx::array(static_cast<int32_t>(acc.card), mx::int32);
	auto zero_i32 = mx::array(static_cast<int32_t>(0), mx::int32);
	auto in_range = mx::logical_and(mx::greater_equal(codes, zero_i32), mx::less(codes, dump));
	auto safe_codes = mx::where(in_range, codes, dump);
	auto nprogs_arr = mx::array(static_cast<int32_t>(nprogs), mx::int32);

	auto row_up = mx::expand_dims(mx::astype(in_range, mx::int32), 1);
	acc.rows = mx::scatter_add(acc.rows, safe_codes, row_up, 0);

	std::vector<mx::array> to_eval = {acc.rows};
	for (size_t p = 0; p < nprogs; p++) {
		auto kind = programs[p].kind;
		if (kind == MlxAggKind::COUNT_STAR) {
			continue;
		}
		auto linear =
		    mx::add(mx::multiply(safe_codes, nprogs_arr), mx::array(static_cast<int32_t>(p), mx::int32));

		if (kind == MlxAggKind::COUNT || !seg.exprs[p].has_value()) {
			auto pass = mx::astype(in_range, mx::int32);
			if (seg.masks[p].has_value()) {
				pass = mx::where(*seg.masks[p], pass, zero_i32);
			}
			acc.counts = mx::scatter_add(acc.counts, linear, mx::expand_dims(pass, 1), 0);
			to_eval.push_back(acc.counts);
			continue;
		}
		if (kind != MlxAggKind::SUM && kind != MlxAggKind::AVG) {
			continue;
		}
		if (programs[p].int_lane) {
			auto val = *seg.exprs[p];
			if (seg.masks[p].has_value()) {
				val = mx::where(*seg.masks[p], val, mx::zeros_like(val));
			}
			val = mx::where(in_range, val, mx::zeros_like(val));
			if (programs[p].int32_rows) {
				acc.isums = mx::scatter_add(acc.isums, linear, mx::expand_dims(val, 1), 0);
				to_eval.push_back(acc.isums);
			} else if (programs[p].gpu_fp32_rows) {
				acc.fsums = mx::scatter_add(acc.fsums, linear, mx::expand_dims(val, 1), 0);
				to_eval.push_back(acc.fsums);
			}
			auto cnt = mx::where(in_range, mx::ones_like(codes), mx::zeros_like(codes));
			if (seg.masks[p].has_value()) {
				cnt = mx::where(*seg.masks[p], cnt, mx::zeros_like(codes));
			}
			acc.counts = mx::scatter_add(acc.counts, linear, mx::expand_dims(mx::astype(cnt, mx::int32), 1), 0);
			to_eval.push_back(acc.counts);
			continue;
		}
		auto val = *seg.exprs[p];
		if (seg.masks[p].has_value()) {
			val = mx::where(*seg.masks[p], val, mx::zeros_like(val));
		}
		val = mx::where(in_range, val, mx::zeros_like(val));
		val = mx::astype(val, mx::float32);
		acc.fsums = mx::scatter_add(acc.fsums, linear, mx::expand_dims(val, 1), 0);
		auto cnt = mx::where(in_range, mx::ones_like(codes), mx::zeros_like(codes));
		if (seg.masks[p].has_value()) {
			cnt = mx::where(*seg.masks[p], cnt, mx::zeros_like(codes));
		}
		acc.counts = mx::scatter_add(acc.counts, linear, mx::expand_dims(mx::astype(cnt, mx::int32), 1), 0);
		to_eval.push_back(acc.fsums);
		to_eval.push_back(acc.counts);
	}
	mx::eval(to_eval);
}

void GroupedGpuDownload(MlxGroupedState &state, GroupedGpuAcc &acc, const std::vector<MlxSumProgram> &programs) {
	GroupedGpuFlushPartials(state, acc, programs);
	mx::eval({acc.rows, acc.counts, acc.fsums});
	auto rows_ptr = acc.rows.data<int32_t>();
	auto counts_ptr = acc.counts.data<int32_t>();
	auto fsums_ptr = acc.fsums.data<float>();
	for (int64_t g = 0; g < acc.card; g++) {
		state.rows[static_cast<size_t>(g)] += rows_ptr[g];
		for (size_t p = 0; p < acc.nprogs; p++) {
			auto slot = static_cast<size_t>(g) * acc.nprogs + p;
			state.counts[slot] += counts_ptr[slot];
			if (!programs[p].int_lane) {
				state.fvalues[slot] += static_cast<double>(fsums_ptr[slot]);
			}
		}
	}
}

MlxGroupedState MlxGroupedInitLike(const MlxGroupedState &state, const std::vector<MlxSumProgram> &programs);

void GroupedAccumulateArrays(MlxGroupedState &state, const GroupedSegmentArrays &seg,
                             const std::vector<MlxSumProgram> &programs) {
	if (state.dense_reduce) {
		GroupedDenseGpuReduce(state, seg, programs);
		return;
	}
	if (state.gpu_handle >= 0) {
		if (auto acc = GroupedGpuLookup(state.gpu_handle)) {
			auto chunk_rows = GpuScatterChunkRows(programs);
			if (chunk_rows >= seg.count) {
				GroupedGpuScatter(*acc, seg, programs);
				GroupedGpuFlushPartials(state, *acc, programs);
			} else {
				for (int64_t off = 0; off < seg.count; off += chunk_rows) {
					auto end = std::min(off + chunk_rows, seg.count);
					auto slice = SliceGroupedSegment(seg, off, end);
					GroupedGpuScatter(*acc, slice, programs);
					GroupedGpuFlushPartials(state, *acc, programs);
				}
			}
			GroupedCpuScatterInt(state, seg, programs);
		}
		return;
	}
	auto n = seg.count;
	if (n == 0) {
		return;
	}
	auto card = state.card;
	auto nprogs = programs.size();
	auto codes = seg.codes.data<int32_t>();
	std::vector<const float *> fexpr(nprogs, nullptr);
	std::vector<const int64_t *> iexpr(nprogs, nullptr);
	std::vector<const uint8_t *> pmask(nprogs, nullptr);
	for (size_t p = 0; p < nprogs; p++) {
		if (seg.exprs[p].has_value()) {
			if (programs[p].int_lane) {
				iexpr[p] = seg.exprs[p]->data<int64_t>();
			} else {
				fexpr[p] = seg.exprs[p]->data<float>();
			}
		}
		if (seg.masks[p].has_value()) {
			pmask[p] = seg.masks[p]->data<uint8_t>();
		}
	}

	auto nthreads = std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), 8));
	std::vector<MlxGroupedState> locals(nthreads);
	for (auto &local : locals) {
		local = MlxGroupedInitLike(state, programs);
	}
	std::vector<const int32_t *> i32expr(nprogs, nullptr);
	for (size_t p = 0; p < nprogs; p++) {
		if (seg.exprs[p].has_value() && programs[p].int_lane && programs[p].int32_rows) {
			i32expr[p] = seg.exprs[p]->data<int32_t>();
			iexpr[p] = nullptr;
		}
	}
	std::vector<std::thread> workers;
	auto chunk = (static_cast<size_t>(n) + nthreads - 1) / nthreads;
	for (size_t t = 0; t < nthreads; t++) {
		auto lo = t * chunk;
		auto hi = std::min<size_t>(lo + chunk, static_cast<size_t>(n));
		if (lo >= hi) {
			break;
		}
		workers.emplace_back([&, lo, hi, t]() {
			auto &local = locals[t];
			// group-existence rows (codes >= card land in a dump slot)
			std::vector<int64_t> rows_dump(static_cast<size_t>(card) + 1, 0);
			for (size_t i = lo; i < hi; i++) {
				auto g = codes[i];
				rows_dump[g < 0 || g > card ? card : g]++;
			}
			for (int64_t g = 0; g < card; g++) {
				local.rows[g] += rows_dump[g];
			}
			// per-program streaming passes: sequential expr reads, one dump
			// accumulator slot instead of a branch per row
			for (size_t p = 0; p < nprogs; p++) {
				auto kind = programs[p].kind;
				if (kind == MlxAggKind::COUNT_STAR) {
					continue;
				}
				auto mask = pmask[p];
				if (kind == MlxAggKind::COUNT || !seg.exprs[p].has_value()) {
					for (size_t i = lo; i < hi; i++) {
						auto g = codes[i];
						if (g < 0 || g >= card || (mask && !mask[i])) {
							continue;
						}
						local.counts[static_cast<size_t>(g) * nprogs + p]++;
					}
					continue;
				}
				if (kind == MlxAggKind::SUM || kind == MlxAggKind::AVG) {
					std::vector<__int128> isum(static_cast<size_t>(card) + 1, 0);
					std::vector<double> fsum(programs[p].int_lane ? 0 : static_cast<size_t>(card) + 1, 0.0);
					std::vector<int64_t> cnt(static_cast<size_t>(card) + 1, 0);
					if (programs[p].int_lane && i32expr[p] && !mask) {
						auto e = i32expr[p];
						for (size_t i = lo; i < hi; i++) {
							auto g = codes[i];
							auto slot = (g < 0 || g > card) ? card : g;
							isum[slot] += e[i];
							cnt[slot]++;
						}
					} else if (programs[p].int_lane && iexpr[p] && !mask) {
						auto e = iexpr[p];
						for (size_t i = lo; i < hi; i++) {
							auto g = codes[i];
							auto slot = (g < 0 || g > card) ? card : g;
							isum[slot] += e[i];
							cnt[slot]++;
						}
					} else {
						for (size_t i = lo; i < hi; i++) {
							auto g = codes[i];
							if (g < 0 || g >= card || (mask && !mask[i])) {
								continue;
							}
							if (programs[p].int_lane) {
								isum[g] += i32expr[p] ? static_cast<int64_t>(i32expr[p][i]) : iexpr[p][i];
							} else {
								fsum[g] += static_cast<double>(fexpr[p][i]);
							}
							cnt[g]++;
						}
					}
					for (int64_t g = 0; g < card; g++) {
						auto slot = static_cast<size_t>(g) * nprogs + p;
						local.ivalues[slot] += isum[g];
						if (!programs[p].int_lane) {
							local.fvalues[slot] += fsum[g];
						}
						local.counts[slot] += cnt[g];
					}
					continue;
				}
				// MIN / MAX
				for (size_t i = lo; i < hi; i++) {
					auto g = codes[i];
					if (g < 0 || g >= card || (mask && !mask[i])) {
						continue;
					}
					auto slot = static_cast<size_t>(g) * nprogs + p;
					local.counts[slot]++;
					if (programs[p].int_lane) {
						auto v = i32expr[p] ? static_cast<int64_t>(i32expr[p][i]) : iexpr[p][i];
						if (kind == MlxAggKind::MIN) {
							local.ivalues[slot] = std::min<__int128>(local.ivalues[slot], v);
						} else {
							local.ivalues[slot] = std::max<__int128>(local.ivalues[slot], v);
						}
					} else {
						auto v = static_cast<double>(fexpr[p][i]);
						if (kind == MlxAggKind::MIN) {
							local.fvalues[slot] = std::min(local.fvalues[slot], v);
						} else {
							local.fvalues[slot] = std::max(local.fvalues[slot], v);
						}
					}
				}
			}
		});
	}
	for (auto &w : workers) {
		w.join();
	}
	for (auto &local : locals) {
		for (int64_t g = 0; g < card; g++) {
			state.rows[g] += local.rows[g];
			for (size_t p = 0; p < nprogs; p++) {
				auto slot = static_cast<size_t>(g) * nprogs + p;
				state.counts[slot] += local.counts[slot];
				switch (programs[p].kind) {
				case MlxAggKind::MIN:
					state.ivalues[slot] = std::min(state.ivalues[slot], local.ivalues[slot]);
					state.fvalues[slot] = std::min(state.fvalues[slot], local.fvalues[slot]);
					break;
				case MlxAggKind::MAX:
					state.ivalues[slot] = std::max(state.ivalues[slot], local.ivalues[slot]);
					state.fvalues[slot] = std::max(state.fvalues[slot], local.fvalues[slot]);
					break;
				default:
					state.ivalues[slot] += local.ivalues[slot];
					state.fvalues[slot] += local.fvalues[slot];
					break;
				}
			}
		}
	}
}

} // namespace

MlxGroupedState MlxGroupedInit(const MlxGroupedSpec &spec, const std::vector<MlxSumProgram> &programs) {
	MlxGroupedState state;
	state.card = 1;
	for (auto c : spec.key_cards) {
		state.card *= c;
	}
	state.nprograms = programs.size();
	auto slots = static_cast<size_t>(state.card) * state.nprograms;
	state.fvalues.assign(slots, 0.0);
	state.ivalues.assign(slots, 0);
	state.counts.assign(slots, 0);
	state.rows.assign(static_cast<size_t>(state.card), 0);
	for (size_t p = 0; p < programs.size(); p++) {
		for (int64_t g = 0; g < state.card; g++) {
			auto slot = static_cast<size_t>(g) * state.nprograms + p;
			if (programs[p].kind == MlxAggKind::MIN) {
				state.fvalues[slot] = std::numeric_limits<double>::infinity();
				state.ivalues[slot] = std::numeric_limits<int64_t>::max();
			} else if (programs[p].kind == MlxAggKind::MAX) {
				state.fvalues[slot] = -std::numeric_limits<double>::infinity();
				state.ivalues[slot] = std::numeric_limits<int64_t>::min();
			}
		}
	}
	state.dense_reduce = state.card > 0 && state.card <= kDenseReduceCard;
	state.gpu_handle = state.dense_reduce ? -1 : GroupedGpuOpen(state, programs);
	return state;
}

void MlxGroupedGpuFinish(MlxGroupedState &state, const std::vector<MlxSumProgram> &programs) {
	if (state.gpu_handle < 0) {
		return;
	}
	if (auto acc = GroupedGpuLookup(state.gpu_handle)) {
		GroupedGpuDownload(state, *acc, programs);
	}
	GroupedGpuClose(state.gpu_handle);
	state.gpu_handle = -1;
}

namespace {
MlxGroupedState MlxGroupedInitLike(const MlxGroupedState &state, const std::vector<MlxSumProgram> &programs) {
	MlxGroupedSpec spec;
	spec.key_cards = {state.card};
	spec.key_cols = {0};
	spec.key_offsets = {0};
	return MlxGroupedInit(spec, programs);
}
} // namespace

void MlxGroupedAccumulate(MlxGroupedState &state, const MlxGroupedSpec &spec, const std::vector<MlxColumnData> &cols,
                          size_t count, const std::vector<MlxSumProgram> &programs, const MlxFilter &filter) {
	if (count == 0) {
		return;
	}
	auto segment = SegmentFromHost(cols, count);
	auto arrays = GroupedEvalSegment(segment, spec, programs, filter, state.dense_reduce);
	GroupedAccumulateArrays(state, arrays, programs);
}

void MlxGroupedAccumulateCached(MlxGroupedState &state, const MlxGroupedSpec &spec,
                                const std::vector<std::string> &col_keys,
                                const std::vector<MlxSumProgram> &programs, const MlxFilter &filter,
                                const std::string &table_prefix) {
	std::string prefix = table_prefix;
	if (prefix.empty() && !col_keys.empty()) {
		auto pos = col_keys[0].rfind('#');
		if (pos != std::string::npos) {
			prefix = col_keys[0].substr(0, pos + 1);
		}
	}
	if (state.dense_reduce && !prefix.empty() && TryLineitemQ1FastPath(state, prefix, col_keys, programs, filter)) {
		return;
	}
	if (state.dense_reduce) {
		MlxSegment mega;
		{
			std::lock_guard<std::mutex> guard(cache_mutex);
			auto &store = CacheStore();
			bool all_fused = true;
			for (auto &key : col_keys) {
				auto it = store.find(key);
				if (it == store.end() || !it->second.fused_col.has_value()) {
					all_fused = false;
					break;
				}
			}
			if (all_fused) {
				mega.count = store[col_keys[0]].rows;
				for (auto &key : col_keys) {
					auto &cached = store[key];
					mega.cols.push_back(*cached.fused_col);
					if (cached.fused_valid.has_value()) {
						mega.valids.push_back(*cached.fused_valid);
					} else {
						mega.valids.push_back(std::nullopt);
					}
				}
			}
		}
		if (mega.count > 0 && !mega.cols.empty()) {
			auto arrays = GroupedEvalSegment(mega, spec, programs, filter, true, true);
			if (GroupedTileKernelEligible(programs, state.card)) {
				std::vector<std::pair<size_t, mx::array>> val_exprs;
				std::vector<std::optional<mx::array>> val_masks;
				val_exprs.reserve(programs.size());
				val_masks.reserve(programs.size());
				for (size_t p = 0; p < programs.size(); p++) {
					auto kind = programs[p].kind;
					if ((kind == MlxAggKind::SUM || kind == MlxAggKind::AVG) && programs[p].int_lane &&
					    arrays.exprs[p].has_value()) {
						val_exprs.push_back({p, *arrays.exprs[p]});
						val_masks.push_back(arrays.masks[p]);
					}
				}
				if (!val_exprs.empty()) {
					std::vector<mx::array> graph = {arrays.codes};
					for (auto &ve : val_exprs) {
						graph.push_back(ve.second);
					}
					for (auto &m : val_masks) {
						if (m.has_value()) {
							graph.push_back(*m);
						}
					}
					mx::eval(graph);
					try {
						GroupedTileKernelAccumulate(state, arrays.codes, val_exprs, val_masks, programs);
						return;
					} catch (...) {
						// Custom Metal kernel unavailable or unsupported layout — dense reduce fallback.
					}
				}
			}
			std::vector<mx::array> graph;
			graph.push_back(arrays.codes);
			for (auto &e : arrays.exprs) {
				if (e.has_value()) {
					graph.push_back(*e);
				}
			}
			for (auto &m : arrays.masks) {
				if (m.has_value()) {
					graph.push_back(*m);
				}
			}
			std::vector<mx::array> reduce_outs;
			GroupedDenseGpuReduce(state, arrays, programs, true, &reduce_outs);
			graph.insert(graph.end(), reduce_outs.begin(), reduce_outs.end());
			mx::eval(graph);
			GroupedDenseGpuReduceMerge(state, arrays, programs, reduce_outs);
			return;
		}
	}

	std::vector<MlxSegment> segments;
	std::vector<std::vector<MlxZoneMap>> segment_zone_maps;
	std::vector<SegmentPrunePred> prune_preds;
	BuildSegmentPrunePreds(filter, prune_preds);
	{
		std::lock_guard<std::mutex> guard(cache_mutex);
		auto &store = CacheStore();
		size_t nsegments = 0;
		segment_zone_maps.resize(col_keys.size());
		for (size_t c = 0; c < col_keys.size(); c++) {
			auto &key = col_keys[c];
			auto it = store.find(key);
			if (it == store.end()) {
				throw std::runtime_error("GPU cache miss for grouped column '" + key + "'");
			}
			segment_zone_maps[c] = it->second.zone_maps;
			if (nsegments == 0) {
				nsegments = it->second.segments.size();
				segments.resize(nsegments);
				for (size_t s = 0; s < nsegments; s++) {
					segments[s].count = static_cast<int64_t>(it->second.segments[s].shape(0));
				}
			} else if (it->second.segments.size() != nsegments) {
				throw std::runtime_error("GPU cache segment mismatch for grouped column '" + key + "'");
			}
			for (size_t s = 0; s < nsegments; s++) {
				segments[s].cols.push_back(it->second.segments[s]);
				segments[s].valids.push_back(it->second.valids[s]);
			}
		}
	}
	size_t seg_idx = 0;
	if (state.dense_reduce) {
		size_t batch_begin = 0;
		int64_t batch_rows = 0;
		auto flush_batch = [&]() {
			if (batch_rows <= 0) {
				return;
			}
			auto batch = ConcatSegmentParts(segments, batch_begin, seg_idx);
			if (batch.count > 0) {
				auto arrays = GroupedEvalSegment(batch, spec, programs, filter, true);
				GroupedDenseGpuReduce(state, arrays, programs);
			}
			batch_begin = seg_idx;
			batch_rows = 0;
		};
		for (size_t s = 0; s < segments.size(); s++, seg_idx++) {
			auto &segment = segments[s];
			if (segment.count > 0 && !SegmentShouldPrune(s, prune_preds, segment_zone_maps)) {
				if (batch_rows == 0) {
					batch_begin = s;
				}
				if (batch_rows > 0 && batch_rows + segment.count > kDenseCachedBatchRows) {
					flush_batch();
					batch_begin = s;
				}
				batch_rows += segment.count;
			} else if (batch_rows > 0) {
				flush_batch();
			}
		}
		flush_batch();
		return;
	}
	seg_idx = 0;
	for (auto &segment : segments) {
		if (segment.count > 0 && !SegmentShouldPrune(seg_idx, prune_preds, segment_zone_maps)) {
			auto arrays = GroupedEvalSegment(segment, spec, programs, filter, false);
			GroupedAccumulateArrays(state, arrays, programs);
		}
		seg_idx++;
	}
}

//===--------------------------------------------------------------------===//
// Dictionary encoding for VARCHAR group keys (cached as fp32 codes)
//===--------------------------------------------------------------------===//

namespace {
struct MlxDictEntry {
	int64_t population = -1;
	std::vector<std::string> strings;
	std::unordered_map<std::string, int32_t> index;
};

std::mutex &DictMutex() {
	static std::mutex m;
	return m;
}

std::unordered_map<std::string, MlxDictEntry> &DictStore() {
	static std::unordered_map<std::string, MlxDictEntry> store;
	return store;
}
} // namespace

int32_t MlxDictEncode(const std::string &col_key, int64_t population, const std::string &value, int64_t max_card) {
	std::lock_guard<std::mutex> guard(DictMutex());
	auto &entry = DictStore()[col_key];
	if (entry.population != population) {
		entry.population = population;
		entry.strings.clear();
		entry.index.clear();
	}
	auto it = entry.index.find(value);
	if (it != entry.index.end()) {
		return it->second;
	}
	if (static_cast<int64_t>(entry.strings.size()) >= max_card) {
		return -1;
	}
	auto code = static_cast<int32_t>(entry.strings.size());
	entry.strings.push_back(value);
	entry.index.emplace(value, code);
	return code;
}

std::vector<std::string> MlxDictStrings(const std::string &col_key, int64_t population) {
	std::lock_guard<std::mutex> guard(DictMutex());
	auto it = DictStore().find(col_key);
	if (it == DictStore().end() || it->second.population != population) {
		return {};
	}
	return it->second.strings;
}

void MlxDictInstall(const std::string &col_key, int64_t population, std::vector<std::string> strings) {
	std::lock_guard<std::mutex> guard(DictMutex());
	auto &entry = DictStore()[col_key];
	entry.population = population;
	entry.index.clear();
	for (size_t i = 0; i < strings.size(); i++) {
		entry.index.emplace(strings[i], static_cast<int32_t>(i));
	}
	entry.strings = std::move(strings);
}

int64_t MlxDictCard(const std::string &col_key, int64_t population) {
	std::lock_guard<std::mutex> guard(DictMutex());
	auto it = DictStore().find(col_key);
	if (it == DictStore().end() || it->second.population != population) {
		return -1;
	}
	return static_cast<int64_t>(it->second.strings.size());
}

int64_t MlxCachePopulation(const std::string &col_key) {
	std::lock_guard<std::mutex> guard(cache_mutex);
	auto it = CacheStore().find(col_key);
	return it == CacheStore().end() ? -1 : it->second.population;
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

std::optional<mx::array> ResolveFusedColumnUnlocked(const std::string &col_key) {
	auto &store = CacheStore();
	auto it = store.find(col_key);
	if (it == store.end() || !it->second.fused_col.has_value()) {
		for (auto &entry : store) {
			if (entry.first.size() >= col_key.size() &&
			    entry.first.compare(entry.first.size() - col_key.size(), col_key.size(), col_key) == 0 &&
			    entry.second.fused_col.has_value()) {
				return *entry.second.fused_col;
			}
		}
		return std::nullopt;
	}
	return *it->second.fused_col;
}

std::optional<mx::array> ResolveFusedColumn(const std::string &col_key) {
	std::lock_guard<std::mutex> guard(cache_mutex);
	return ResolveFusedColumnUnlocked(col_key);
}

std::string MlxStreamSumBench(const std::string &col_key) {
	auto col = ResolveFusedColumn(col_key);
	if (!col.has_value()) {
		size_t fused = 0;
		{
			std::lock_guard<std::mutex> guard(cache_mutex);
			for (auto &entry : CacheStore()) {
				if (entry.second.fused_col.has_value()) {
					fused++;
				}
			}
		}
		return "error=missing fused column '" + col_key + "' (fused_cols=" + std::to_string(fused) +
		       ", use suffix like lineitem#5)";
	}
	mx::array values = col->dtype() == mx::int64 ? *col : mx::astype(*col, mx::int64);
	auto n = values.shape(0);
	auto t0 = std::chrono::steady_clock::now();
	auto sum = StreamingInt64Sum(values);
	auto t1 = std::chrono::steady_clock::now();
	double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	double gib_s = ms > 0 ? (static_cast<double>(n) * 8.0) / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
	return "sum=" + std::to_string(sum) + " rows=" + std::to_string(n) + " ms=" + std::to_string(ms) +
	       " gib_s=" + std::to_string(gib_s);
}

std::string MlxMultiAggBench(const std::string &col_key) {
	auto col = ResolveFusedColumn(col_key);
	if (!col.has_value()) {
		return "error=missing fused column '" + col_key + "'";
	}
	mx::array values = col->dtype() == mx::int64 ? *col : mx::astype(*col, mx::int64);
	auto n = values.shape(0);
	auto t0 = std::chrono::steady_clock::now();
	auto agg = StreamingMultiAgg(values);
	auto t1 = std::chrono::steady_clock::now();
	double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	double gib_s = ms > 0 ? (static_cast<double>(n) * 8.0) / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
	return "sum=" + std::to_string(agg.sum) + " min=" + std::to_string(agg.min) + " max=" + std::to_string(agg.max) +
	       " count=" + std::to_string(agg.count) + " rows=" + std::to_string(n) + " ms=" + std::to_string(ms) +
	       " gib_s=" + std::to_string(gib_s);
}

} // namespace duckdb_mlx
