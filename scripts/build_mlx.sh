#!/usr/bin/env bash
# One-time build of the vendored MLX (third_party/mlx) into build/mlx-install,
# where the extension's CMake picks it up with find_package(MLX). Invoked
# automatically from CMakeLists.txt when the install prefix is missing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/mlx"
BUILD_DIR="$ROOT/build/mlx-build"
INSTALL_DIR="$ROOT/build/mlx-install"

if [ ! -f "$SRC/CMakeLists.txt" ]; then
	echo "third_party/mlx is empty — run: git submodule update --init third_party/mlx" >&2
	exit 1
fi

cmake -S "$SRC" -B "$BUILD_DIR" -G Ninja \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
	-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	-DMLX_BUILD_TESTS=OFF \
	-DMLX_BUILD_EXAMPLES=OFF \
	-DMLX_BUILD_BENCHMARKS=OFF \
	-DMLX_BUILD_PYTHON_BINDINGS=OFF
cmake --build "$BUILD_DIR" -j
cmake --install "$BUILD_DIR"
