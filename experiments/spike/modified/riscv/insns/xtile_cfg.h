// TILE-ENGINE-MODIFICATION-START

// This instruction consumes one 64-bit configuration value.
require_rv64;

const reg_t cfg = RS1;

// [63:38] = RESERVED
// [37:28] = K --> Reduction Length
// [27:18] = N --> Number of output column
// [17:8] = M --> Number of output rows
// [7:5] = TILE_COLUMNS_MINUS_1 --> active columns = field + 1
// [4:2] = TILE_ROWS_MINUS_1 --> active rows = field + 1
// [1] = B_TYPE --> 0: signed INT8 ; 1: signed INT32
// [0] = A_TYPE --> 0: signed INT8 ; 1: signed INT32

// Matrix dimensions are stored directly as 10-bit unsigned values.
const reg_t m = (cfg >> 8)  & 0x3ff; // bit [17:8]
const reg_t n = (cfg >> 18) & 0x3ff; // bit [27:18]
const reg_t k = (cfg >> 28) & 0x3ff; // bit [37:28]

// Validate before modifying the retained configuration.
require((cfg >> 38) == 0);
require(m != 0);
require(n != 0);
require(k != 0);

// Retain the configuration for subsequent tile instructions.
STATE.xtile_cfg = cfg;
STATE.xtile_cfg_valid = true;

// TILE-ENGINE-MODIFICATION-END
