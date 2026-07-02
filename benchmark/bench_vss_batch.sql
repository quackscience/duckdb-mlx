-- Batched vector search: Q queries x top-10 over 1M x 384 fp32.
-- CPU: LATERAL brute force (scales linearly with Q)
-- GPU: one (Q x 384) @ (384 x 1M) matmul against the pinned matrix
.timer on

CREATE TABLE items AS
SELECT i AS id, list_transform(range(384), j -> random()::FLOAT)::FLOAT[384] AS emb
FROM range(1000000) t(i);

SELECT '=== pin (cold, one-time) ===' AS bench;
SELECT mlx_vss_pin('items', list(emb ORDER BY id)) FROM items;

CREATE TABLE queries AS SELECT id AS qid, emb FROM items WHERE id % 31250 = 17; -- 32 queries
SET VARIABLE qs32 = (SELECT list(emb::FLOAT[] ORDER BY qid) FROM queries);

SELECT '=== cpu 32 queries (lateral) ===' AS bench;
SELECT count(*) FROM (
  SELECT qid, id FROM queries, LATERAL (
    SELECT id FROM items ORDER BY array_cosine_similarity(items.emb, queries.emb) DESC LIMIT 10));
SELECT count(*) FROM (
  SELECT qid, id FROM queries, LATERAL (
    SELECT id FROM items ORDER BY array_cosine_similarity(items.emb, queries.emb) DESC LIMIT 10));

SELECT '=== gpu 32 queries (one matmul) ===' AS bench;
SELECT count(*) FROM mlx_vss_search_batch('items', getvariable('qs32'), 10);
SELECT count(*) FROM mlx_vss_search_batch('items', getvariable('qs32'), 10);
SELECT count(*) FROM mlx_vss_search_batch('items', getvariable('qs32'), 10);

CREATE TABLE queries128 AS SELECT id AS qid, emb FROM items WHERE id % 7812 = 17; -- 128 queries
SET VARIABLE qs128 = (SELECT list(emb::FLOAT[] ORDER BY qid) FROM queries128);

SELECT '=== cpu 128 queries (lateral) ===' AS bench;
SELECT count(*) FROM (
  SELECT qid, id FROM queries128, LATERAL (
    SELECT id FROM items ORDER BY array_cosine_similarity(items.emb, queries128.emb) DESC LIMIT 10));

SELECT '=== gpu 128 queries (one matmul) ===' AS bench;
SELECT count(*) FROM mlx_vss_search_batch('items', getvariable('qs128'), 10);
SELECT count(*) FROM mlx_vss_search_batch('items', getvariable('qs128'), 10);
SELECT count(*) FROM mlx_vss_search_batch('items', getvariable('qs128'), 10);
