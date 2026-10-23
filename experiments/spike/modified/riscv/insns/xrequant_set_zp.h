// REQUANT-ENGINE-MODIFICATION-START

require_rv64;

const int32_t input_zp =
    static_cast<int32_t>(static_cast<uint32_t>(RS1));

const int32_t output_zp =
    static_cast<int32_t>(static_cast<uint32_t>(RS2));

STATE.xrequant_input_zp = input_zp;
STATE.xrequant_output_zp = output_zp;
STATE.xrequant_zp_valid = true;

// REQUANT-ENGINE-MODIFICATION-END