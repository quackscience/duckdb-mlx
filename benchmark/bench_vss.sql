-- Vector similarity search: DuckDB CPU brute force vs GPU-resident matrix.
-- 1M vectors x 384 dims fp32 (~1.5 GB), typical sentence-embedding scale.
-- pin: one-time GPU cache build (reported separately, GQE methodology)
.timer on

SELECT '=== generate 1M x 384 ===' AS bench;
CREATE TABLE items AS
SELECT i AS id, list_transform(range(384), j -> random()::FLOAT)::FLOAT[384] AS emb
FROM range(1000000) t(i);

SELECT '=== pin (cold, one-time) ===' AS bench;
SELECT mlx_vss_pin('items', list(emb ORDER BY id)) FROM items;

SELECT '=== query vector ===' AS bench;
SET VARIABLE q = (SELECT emb::FLOAT[] FROM items WHERE id = 500000);

SELECT '=== cpu brute force top-10 ===' AS bench;
SELECT id FROM items ORDER BY array_cosine_similarity(emb, getvariable('q')::FLOAT[384]) DESC LIMIT 10;
SELECT id FROM items ORDER BY array_cosine_similarity(emb, getvariable('q')::FLOAT[384]) DESC LIMIT 10;
SELECT id FROM items ORDER BY array_cosine_similarity(emb, getvariable('q')::FLOAT[384]) DESC LIMIT 10;
SELECT id FROM items ORDER BY array_cosine_similarity(emb, getvariable('q')::FLOAT[384]) DESC LIMIT 10;
SELECT id FROM items ORDER BY array_cosine_similarity(emb, getvariable('q')::FLOAT[384]) DESC LIMIT 10;

SELECT '=== gpu top-10 ===' AS bench;
SELECT idx, score FROM mlx_vss_search('items', getvariable('q'), 10);
SELECT idx, score FROM mlx_vss_search('items', getvariable('q'), 10);
SELECT idx, score FROM mlx_vss_search('items', getvariable('q'), 10);
SELECT idx, score FROM mlx_vss_search('items', getvariable('q'), 10);
SELECT idx, score FROM mlx_vss_search('items', getvariable('q'), 10);

SELECT '=== sanity: both find id=500000 first ===' AS bench;
SELECT (SELECT idx FROM mlx_vss_search('items', getvariable('q'), 1)) =
       (SELECT id FROM items ORDER BY array_cosine_similarity(emb, getvariable('q')::FLOAT[384]) DESC LIMIT 1)
       AS agree;
