cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME duckdb_mlx)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package.
find_package(spdlog CONFIG REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD "17" CACHE STRING "C++ standard to enforce")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The Metal/MLX GPU execution path only exists on Apple Silicon. On every
# other platform the extension still builds and loads, but declines all plans
# so queries stay on the DuckDB CPU engine.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  set(DUCKDB_MLX_GPU_DEFAULT ON)
else()
  set(DUCKDB_MLX_GPU_DEFAULT OFF)
endif()
option(DUCKDB_MLX_ENABLE_GPU "Build the Metal/MLX GPU execution path" ${DUCKDB_MLX_GPU_DEFAULT})

# MLX is Apple-only and not in vcpkg; it is vendored as a submodule and built
# once into build/mlx-install, then consumed as an imported target.
if(DUCKDB_MLX_ENABLE_GPU)
  set(DUCKDB_MLX_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/mlx-install)
  if(NOT EXISTS ${DUCKDB_MLX_INSTALL_DIR})
    message(STATUS "Building vendored MLX into ${DUCKDB_MLX_INSTALL_DIR} (one-time)")
    execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_mlx.sh RESULT_VARIABLE DUCKDB_MLX_BUILD_RESULT)
    if(NOT DUCKDB_MLX_BUILD_RESULT EQUAL 0)
      message(FATAL_ERROR "Vendored MLX build failed — run ./scripts/build_mlx.sh for details")
    endif()
  endif()
  find_package(MLX CONFIG REQUIRED PATHS ${DUCKDB_MLX_INSTALL_DIR} NO_DEFAULT_PATH)
endif()

include_directories(src/include)

set(EXTENSION_SOURCES src/duckdb_mlx_extension.cpp src/ops/mlx_vss.cpp)
if(DUCKDB_MLX_ENABLE_GPU)
  list(APPEND EXTENSION_SOURCES src/planner/mlx_transparent.cpp)
endif()

# mlx_logger.cpp isolates spdlog: DuckDB's vendored fmt shadows the real fmt on
# the include path (same <fmt/...> paths, renamed duckdb_fmt namespace), which
# breaks spdlog. Compile it as an object library with the inherited DuckDB
# include directories cleared so only spdlog's own fmt is visible. It is a
# STATIC library (not OBJECT) because the static extension is exported through
# DuckDB's export set, which every linked in-build target must join.
add_library(duckdb_mlx_logger STATIC src/mlx_logger.cpp)
set_target_properties(
  duckdb_mlx_logger PROPERTIES INCLUDE_DIRECTORIES "" CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                               POSITION_INDEPENDENT_CODE ON)
target_include_directories(duckdb_mlx_logger PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
target_link_libraries(duckdb_mlx_logger PRIVATE spdlog::spdlog)

# src/bridge/ isolates MLX the same way: MLX headers and DuckDB headers never
# meet in one translation unit; data crosses as raw pointers (unified memory).
if(DUCKDB_MLX_ENABLE_GPU)
  add_library(duckdb_mlx_bridge STATIC src/bridge/mlx_bridge.cpp src/bridge/mlx_groupby.cpp src/bridge/mlx_grouped_kernel.cpp)
  set_target_properties(
    duckdb_mlx_bridge PROPERTIES INCLUDE_DIRECTORIES "" CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                                 POSITION_INDEPENDENT_CODE ON)
  target_include_directories(duckdb_mlx_bridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
  target_link_libraries(duckdb_mlx_bridge PRIVATE mlx)
endif()

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# DuckDB's build defaults extension targets to C++11; this project (and MLX,
# later) requires C++17.
set_target_properties(${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME} PROPERTIES CXX_STANDARD 17
                                                                              CXX_STANDARD_REQUIRED ON)

target_link_libraries(${EXTENSION_NAME} duckdb_mlx_logger spdlog::spdlog)
target_link_libraries(${LOADABLE_EXTENSION_NAME} duckdb_mlx_logger spdlog::spdlog)

if(DUCKDB_MLX_ENABLE_GPU)
  target_compile_definitions(${EXTENSION_NAME} PRIVATE DUCKDB_MLX_GPU_ENABLED)
  target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE DUCKDB_MLX_GPU_ENABLED)
  target_link_libraries(${EXTENSION_NAME} duckdb_mlx_bridge mlx)
  target_link_libraries(${LOADABLE_EXTENSION_NAME} duckdb_mlx_bridge mlx)
  install(
    TARGETS duckdb_mlx_bridge
    EXPORT "${DUCKDB_EXPORT_SET}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
endif()

install(
  TARGETS ${EXTENSION_NAME} duckdb_mlx_logger
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
