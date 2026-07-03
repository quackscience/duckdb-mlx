-- Minimal CPU vs GPU snapshot (moderate scale, ~1–2 min total).
.timer on

LOAD mlx;
SET mlx_min_rows = 1000;

SELECT '--- 1. expression SUM (10M rows) ---' AS section;

CREATE OR REPLACE TABLE t_expr AS
SELECT ((range * 7919) % 1000003)::DOUBLE AS x FROM range(10000000);

SET mlx_execution = false;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t_expr;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t_expr;

SELECT mlx_cache_clear();
SET mlx_execution = true;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t_expr;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t_expr;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t_expr;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t_expr;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t_expr;

SELECT '--- 2. plain SUM (10M rows) ---' AS section;

SET mlx_execution = false;
SELECT sum(x) FROM t_expr;
SELECT sum(x) FROM t_expr;

SET mlx_execution = true;
SELECT sum(x) FROM t_expr;
SELECT sum(x) FROM t_expr;
SELECT sum(x) FROM t_expr;

SELECT '--- 3. GROUP BY (5M rows, 1000 groups) ---' AS section;

CREATE OR REPLACE TABLE t_gb AS
SELECT (range % 1000)::BIGINT AS k, (range * 1.5)::DOUBLE AS v FROM range(5000000);

SET mlx_execution = false;
SELECT sum(s) FROM (SELECT k, sum(v) AS s FROM t_gb GROUP BY k) q;
SELECT sum(s) FROM (SELECT k, sum(v) AS s FROM t_gb GROUP BY k) q;

SELECT mlx_cache_clear();
SET mlx_execution = true;
SELECT sum(s) FROM (SELECT k, sum(v) AS s FROM t_gb GROUP BY k) q;
SELECT sum(s) FROM (SELECT k, sum(v) AS s FROM t_gb GROUP BY k) q;
SELECT sum(s) FROM (SELECT k, sum(v) AS s FROM t_gb GROUP BY k) q;
SELECT sum(s) FROM (SELECT k, sum(v) AS s FROM t_gb GROUP BY k) q;
SELECT sum(s) FROM (SELECT k, sum(v) AS s FROM t_gb GROUP BY k) q;

SELECT '--- 4. filtered multi-agg (10M rows, ~90% selectivity) ---' AS section;

SET mlx_execution = false;
SELECT count(*), sum(x) FROM t_expr WHERE x > 100000.5;
SELECT count(*), sum(x) FROM t_expr WHERE x > 100000.5;

SET mlx_execution = true;
SELECT count(*), sum(x) FROM t_expr WHERE x > 100000.5;
SELECT count(*), sum(x) FROM t_expr WHERE x > 100000.5;
SELECT count(*), sum(x) FROM t_expr WHERE x > 100000.5;

SELECT '--- 5. Q6-shaped selective expression SUM (2M rows, ~2% pass) ---' AS section;

CREATE OR REPLACE TABLE t_q6 AS
SELECT
    (range % 50 + 1)::DOUBLE AS l_quantity,
    (range * 0.001)::DOUBLE AS l_discount,
    (1994 + (range % 365))::DOUBLE AS l_shipdate_ord
FROM range(2000000);

SET mlx_execution = false;
SELECT sum(l_quantity * l_discount) FROM t_q6
WHERE l_quantity < 24.5 AND l_shipdate_ord >= 2191.5 AND l_shipdate_ord < 2556.5;
SELECT sum(l_quantity * l_discount) FROM t_q6
WHERE l_quantity < 24.5 AND l_shipdate_ord >= 2191.5 AND l_shipdate_ord < 2556.5;

SELECT mlx_cache_clear();
SET mlx_execution = true;
SELECT sum(l_quantity * l_discount) FROM t_q6
WHERE l_quantity < 24.5 AND l_shipdate_ord >= 2191.5 AND l_shipdate_ord < 2556.5;
SELECT sum(l_quantity * l_discount) FROM t_q6
WHERE l_quantity < 24.5 AND l_shipdate_ord >= 2191.5 AND l_shipdate_ord < 2556.5;
SELECT sum(l_quantity * l_discount) FROM t_q6
WHERE l_quantity < 24.5 AND l_shipdate_ord >= 2191.5 AND l_shipdate_ord < 2556.5;
SELECT sum(l_quantity * l_discount) FROM t_q6
WHERE l_quantity < 24.5 AND l_shipdate_ord >= 2191.5 AND l_shipdate_ord < 2556.5;
SELECT sum(l_quantity * l_discount) FROM t_q6
WHERE l_quantity < 24.5 AND l_shipdate_ord >= 2191.5 AND l_shipdate_ord < 2556.5;
