-- Generate TPC-H at the requested scale factor (passed via -variable sf=N).
LOAD tpch;
CALL dbgen(sf := getvariable('sf'));
