PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=mlx
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Bootstrap vcpkg (spdlog via vcpkg.json) for local builds unless CI sets
# VCPKG_TOOLCHAIN_PATH to a different installation.
include extension-ci-tools/makefiles/vcpkg.Makefile
VCPKG_ROOT := $(PROJ_DIR)vcpkg
export VCPKG_TOOLCHAIN_PATH ?= $(VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake
VCPKG_OVERLAY_PORTS := $(PROJ_DIR)extension-ci-tools/vcpkg_ports

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    VCPKG_DEFAULT_TRIPLET ?= arm64-osx
  else
    VCPKG_DEFAULT_TRIPLET ?= x64-osx
  endif
else ifeq ($(UNAME_M),aarch64)
  VCPKG_DEFAULT_TRIPLET ?= arm64-linux
else
  VCPKG_DEFAULT_TRIPLET ?= x64-linux
endif
export VCPKG_TARGET_TRIPLET ?= $(VCPKG_DEFAULT_TRIPLET)

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

.PHONY: setup-vcpkg-deps fix-stale-vcpkg-cache

# Install manifest deps before cmake. A first configure without
# VCPKG_MANIFEST_DIR poisons CMakeCache (manifest mode OFF); fix-stale-vcpkg-cache
# removes those build dirs so the next configure runs vcpkg install correctly.
setup-vcpkg-deps: setup-vcpkg
	@if [ ! -d "$(VCPKG_OVERLAY_PORTS)" ]; then \
		echo "Missing extension-ci-tools — run: git submodule update --init --recursive"; \
		exit 1; \
	fi
	cd $(PROJ_DIR) && VCPKG_OVERLAY_PORTS=$(VCPKG_OVERLAY_PORTS) $(VCPKG_ROOT)/vcpkg install --triplet $(VCPKG_DEFAULT_TRIPLET)

fix-stale-vcpkg-cache:
	@for d in $(PROJ_DIR)build/release $(PROJ_DIR)build/debug $(PROJ_DIR)build/reldebug \
		$(PROJ_DIR)build/relassert $(PROJ_DIR)build/tidy; do \
	  if [ -f "$$d/CMakeCache.txt" ] && grep -q 'VCPKG_MANIFEST_MODE:BOOL=OFF' "$$d/CMakeCache.txt" 2>/dev/null; then \
	    echo "Removing stale CMake cache in $$d (configured without vcpkg manifest)."; \
	    rm -rf "$$d"; \
	  fi; \
	done

release debug relassert reldebug test_release_internal test_debug_internal test_reldebug_internal tidy-check clangd: setup-vcpkg-deps fix-stale-vcpkg-cache