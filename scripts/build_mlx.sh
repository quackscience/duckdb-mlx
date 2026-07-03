#!/usr/bin/env bash
# One-time build of the vendored MLX (third_party/mlx) into build/mlx-install,
# where the extension's CMake picks it up with find_package(MLX). Invoked
# automatically from CMakeLists.txt when the install prefix is missing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/mlx"
BUILD_DIR="$ROOT/build/mlx-build"
INSTALL_DIR="$ROOT/build/mlx-install"
LOG="$BUILD_DIR/build.log"

if [ ! -f "$SRC/CMakeLists.txt" ]; then
	echo "third_party/mlx is empty — run: git submodule update --init third_party/mlx" >&2
	exit 1
fi

# Metal-only build: skip CPU backend and JACCL (RDMA/Thunderbolt distributed).
# JACCL is enabled on macOS SDK >= 26.2 when MLX_BUILD_CPU=ON and often fails on
# machines without Apple's RDMA stack; duckdb-mlx only needs the Metal GPU path.
cmake -S "$SRC" -B "$BUILD_DIR" -G Ninja \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
	-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	-DMLX_BUILD_TESTS=OFF \
	-DMLX_BUILD_EXAMPLES=OFF \
	-DMLX_BUILD_BENCHMARKS=OFF \
	-DMLX_BUILD_PYTHON_BINDINGS=OFF \
	-DMLX_BUILD_CPU=OFF \
	-DMLX_BUILD_SAFETENSORS=OFF \
	-DMLX_BUILD_GGUF=OFF

if ! cmake --build "$BUILD_DIR" -j 2>&1 | tee "$LOG"; then
	echo "MLX build failed — last 40 lines of $LOG:" >&2
	tail -40 "$LOG" >&2
	exit 1
fi

cmake --install "$BUILD_DIR"
