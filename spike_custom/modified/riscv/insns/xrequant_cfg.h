// REQUANT-ENGINE-MODIFICATION-START

require_rv64;

const reg_t cfg = RS1;

const reg_t mode = cfg & 0x7;
const reg_t shift = (cfg >> 3) & 0x3f;
const reg_t count = (cfg >> 9) & 0xffffffff;

require((cfg >> 41) == 0);
require(mode <= 4);
require(count != 0);

if (mode == 4) {
  // Second shift; the first Q31 shift is fixed at 31.
  require(shift <= 31);
} else {
  require(shift != 0);
}

// Commit only after successful validation.
STATE.xrequant_cfg = cfg;
STATE.xrequant_cfg_valid = true;

// REQUANT-ENGINE-MODIFICATION-END