-- ALU-dense expression benchmark: sum(sin(x)*cos(x) + sqrt(abs(x)+1))
-- cpu: DuckDB all-core fp64 | gpu: list(x) + bridge + MLX fused fp32
.timer on

SELECT '=== 10M rows ===' AS bench;
CREATE OR REPLACE TABLE t AS SELECT (range * 7919) % 1000003 AS x FROM range(10000000);
SELECT 'warmup' AS run, sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT 'cpu' AS run, sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT 'cpu' AS run, sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT 'cpu' AS run, sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT 'warmup' AS run, mlx_expr_bench(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_expr_bench(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_expr_bench(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_expr_bench(list(x)) FROM t;

SELECT '=== 100M rows ===' AS bench;
CREATE OR REPLACE TABLE t AS SELECT (range * 7919) % 1000003 AS x FROM range(100000000);
SELECT 'warmup' AS run, sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT 'cpu' AS run, sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT 'cpu' AS run, sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT 'cpu' AS run, sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT 'warmup' AS run, mlx_expr_bench(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_expr_bench(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_expr_bench(list(x)) FROM t;
SELECT 'gpu' AS run, mlx_expr_bench(list(x)) FROM t;
