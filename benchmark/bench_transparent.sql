-- Transparent GPU execution with the GPU-resident column cache.
-- Same plain SQL; mlx_execution toggles the engine. First GPU run is cold
-- (scan + cache build, reported separately per GQE methodology); hot runs
-- are served entirely from GPU-resident columns with no table scan.
.timer on

CREATE TABLE t AS SELECT ((range * 7919) % 1000003)::DOUBLE AS x FROM range(100000000);

SELECT '=== gpu cold (scan + cache build) ===' AS bench;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;

SELECT '=== gpu hot (GPU-resident, fused) ===' AS bench;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;

SELECT '=== gpu hot: plain sum over cached column ===' AS bench;
SELECT sum(x) FROM t;
SELECT sum(x) FROM t;
SELECT sum(x) FROM t;

SELECT '=== cpu (mlx_execution = false) ===' AS bench;
SET mlx_execution = false;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT sum(sin(x) * cos(x) + sqrt(abs(x) + 1)) FROM t;
SELECT sum(x) FROM t;
SELECT sum(x) FROM t;
SELECT sum(x) FROM t;
