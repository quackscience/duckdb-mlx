#pragma once

namespace duckdb {

class ExtensionLoader;

//! Registers mlx_vss_pin() and mlx_vss_search() — GPU-resident cosine
//! similarity search over a pinned embedding matrix.
void RegisterMlxVss(ExtensionLoader &loader);

} // namespace duckdb
