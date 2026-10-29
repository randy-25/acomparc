// TILE-ENGINE-MODIFICATION-START

require_rv64;

// Read both source operands before updating engine state.
const reg_t a_base = RS1;
const reg_t bt_base = RS2;

// Retain the addresses for subsequent tile operations.
STATE.xtile_a_base = a_base;
STATE.xtile_bt_base = bt_base;
STATE.xtile_ab_valid = true;

// TILE-ENGINE-MODIFICATION-END