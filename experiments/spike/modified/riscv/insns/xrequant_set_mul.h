// REQUANT-ENGINE-MODIFICATION-START

require_rv64;

const int64_t multiplier = static_cast<int64_t>(RS1);

STATE.xrequant_multiplier = multiplier;
STATE.xrequant_mul_valid = true;

// REQUANT-ENGINE-MODIFICATION-END