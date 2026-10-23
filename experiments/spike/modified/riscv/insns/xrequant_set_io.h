// REQUANT-ENGINE-MODIFICATION-START

require_rv64;

const reg_t src_base = RS1;
const reg_t dst_base = RS2;

STATE.xrequant_src_base = src_base;
STATE.xrequant_dst_base = dst_base;
STATE.xrequant_io_valid = true;

// REQUANT-ENGINE-MODIFICATION-END