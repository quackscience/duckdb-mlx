#!/usr/bin/env bash
# Parse bench_minimal.sql timings into a short report.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build/release}"
DUCKDB="${DUCKDB:-$BUILD/duckdb}"
SQL="$ROOT/benchmark/bench_minimal.sql"
OUT="${TMPDIR:-/tmp}/mlx_bench_minimal_$$.txt"

if [[ ! -x "$DUCKDB" ]]; then
  echo "Build first: GEN=ninja make release" >&2
  exit 1
fi

echo "Running minimal benchmark ($(uname -m), $(sw_vers -productVersion 2>/dev/null || echo unknown))..."
"$DUCKDB" -unsigned < "$SQL" 2>&1 | tee "$OUT"

python3 - "$OUT" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path).read()
sections = re.split(r"--- (\d+\. [^-]+) ---", text)
# sections: [preamble, title1, body1, title2, body2, ...]

def times(block):
    return [float(m.group(1)) for m in re.finditer(r"Run Time \(s\): real ([0-9.]+)", block)]

rows = []
for i in range(1, len(sections), 2):
    title = sections[i].strip()
    block = sections[i + 1] if i + 1 < len(sections) else ""
    ts = times(block)
    if len(ts) < 2:
        continue
    # Layout per section: 2 cpu, then gpu (cold + hot runs)
    if title.startswith("1.") or title.startswith("4.") or title.startswith("5."):
        cpu = sum(ts[0:2]) / 2
        gpu_cold = ts[2] if len(ts) > 2 else None
        hot = ts[3:] if len(ts) > 3 else []
    elif title.startswith("2."):
        cpu = sum(ts[0:2]) / 2
        gpu_cold = None
        hot = ts[2:]
    elif title.startswith("3.") or title.startswith("5."):
        cpu = sum(ts[0:2]) / 2
        gpu_cold = ts[2] if len(ts) > 2 else None
        hot = ts[3:]
    else:
        continue
    gpu_hot = sum(hot) / len(hot) if hot else None
    speedup = (cpu / gpu_hot) if gpu_hot and gpu_hot > 0 else None
    rows.append((title, cpu * 1000, (gpu_cold or 0) * 1000, (gpu_hot or 0) * 1000, speedup))

print()
print("=" * 72)
print(f"{'Workload':<42} {'CPU':>8} {'GPU cold':>10} {'GPU hot':>10} {'Speedup':>8}")
print(f"{'':42} {'(ms)':>8} {'(ms)':>10} {'(ms)':>10} {'hot':>8}")
print("-" * 72)
for title, cpu, cold, hot, sp in rows:
    cold_s = f"{cold:8.1f}" if cold > 0 else "       —"
    hot_s = f"{hot:8.1f}" if hot > 0 else "       —"
    sp_s = f"{sp:7.2f}x" if sp else "      —"
    print(f"{title:<42} {cpu:8.1f} {cold_s:>10} {hot_s:>10} {sp_s:>8}")
print("=" * 72)
print("GPU hot = avg of repeat runs after cache warm. Speedup = CPU / GPU hot.")
PY

rm -f "$OUT"
