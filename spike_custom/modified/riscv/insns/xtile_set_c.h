// TILE-ENGINE-MODIFICATION-START

require_rv64;

const reg_t c_base = RS1;

// Retain the output address for subsequent tile operations.
STATE.xtile_c_base = c_base;
STATE.xtile_c_valid = true;

// TILE-ENGINE-MODIFICATION-END