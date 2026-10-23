// TILE-ENGINE-MODIFICATION-START

require_rv64;

// Keep the low 32 bits and interpret them as signed zero points.
const int32_t a_zp =
    static_cast<int32_t>(static_cast<uint32_t>(RS1));

const int32_t b_zp =
    static_cast<int32_t>(static_cast<uint32_t>(RS2));

// Update state after reading both operands.
STATE.xtile_a_zp = a_zp;
STATE.xtile_b_zp = b_zp;
STATE.xtile_zp_valid = true;

// TILE-ENGINE-MODIFICATION-END