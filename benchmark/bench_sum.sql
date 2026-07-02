-- GPU-vs-CPU reduction benchmark (Phase 0 attribution).
-- cpu:  DuckDB all-core sum (note: 128-bit accumulator)
-- ctl:  list(x) materialization only — overhead the mlx_sum vehicle pays
--       before the GPU sees data (goes away with the optimizer hook)
-- gpu:  list(x) + bridge + MLX GPU reduction
.timer on

SELECT '=== 10M rows ===' AS bench;
CREATE OR REPLACE TABLE t AS SELECT (range * 7919) % 1000003 AS x FROM range(10000000);
SELECT 'warmup' AS run, sum(x) FROM t;
SELECT 'cpu' AS run, sum(x) FROM t;
SELECT 'cpu' AS run, sum(x) FROM t;
SELECT 'cpu' AS run, sum(x) FROM t;
SELECT 'cpu' AS run, sum(x) FROM t;
SELECT 'cpu' AS run, sum(x) FROM t;
SELECT 'ctl' AS run, len(list(x)) FROM t;
SELECT 'ctl' AS run, len(list(x)) FROM t;
SELECT 'warmup' AS run, mlx_sum(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_sum(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_sum(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_sum(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_sum(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_sum(list(x)) FROM t;

SELECT '=== 100M rows ===' AS bench;
CREATE OR REPLACE TABLE t AS SELECT (range * 7919) % 1000003 AS x FROM range(100000000);
SELECT 'warmup' AS run, sum(x) FROM t;
SELECT 'cpu' AS run, sum(x) FROM t;
SELECT 'cpu' AS run, sum(x) FROM t;
SELECT 'cpu' AS run, sum(x) FROM t;
SELECT 'cpu' AS run, sum(x) FROM t;
SELECT 'cpu' AS run, sum(x) FROM t;
SELECT 'ctl' AS run, len(list(x)) FROM t;
SELECT 'ctl' AS run, len(list(x)) FROM t;
SELECT 'warmup' AS run, mlx_sum(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_sum(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_sum(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_sum(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_sum(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_sum(list(x)) FROM t;
