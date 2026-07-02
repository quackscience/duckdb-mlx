-- GROUP BY spike: GPU sort+scatter vs hash Metal kernel vs CPU.
.timer on

CREATE TABLE t AS
SELECT (range % 1000)::BIGINT AS k, (range * 1.5)::DOUBLE AS v
FROM range(5000000);

SELECT '=== cpu group by ===' AS bench;
SET mlx_execution = false;
SELECT sum(x) FROM (SELECT k, sum(v) AS x FROM t GROUP BY k) sub;
SELECT sum(x) FROM (SELECT k, sum(v) AS x FROM t GROUP BY k) sub;

SELECT '=== gpu transparent group by (sort path) ===' AS bench;
SET mlx_execution = true;
SELECT sum(x) FROM (SELECT k, sum(v) AS x FROM t GROUP BY k) sub;
SELECT sum(x) FROM (SELECT k, sum(v) AS x FROM t GROUP BY k) sub;
SELECT sum(x) FROM (SELECT k, sum(v) AS x FROM t GROUP BY k) sub;

SELECT '=== gpu mlx_groupby_bench sort ===' AS bench;
SELECT mlx_groupby_bench(list(k), list(v::BIGINT)) FROM t;
SELECT mlx_groupby_bench(list(k), list(v::BIGINT)) FROM t;

SELECT '=== gpu mlx_groupby_bench hash ===' AS bench;
SELECT mlx_groupby_bench(list(k), list(v::BIGINT), true) FROM t;
SELECT mlx_groupby_bench(list(k), list(v::BIGINT), true) FROM t;
