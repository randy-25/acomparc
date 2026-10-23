// REQUANT-ENGINE-MODIFICATION-START

require_rv64;

require(STATE.xrequant_cfg_valid);
require(STATE.xrequant_io_valid);
require(STATE.xrequant_mul_valid);
require(STATE.xrequant_zp_valid);

const reg_t cfg = STATE.xrequant_cfg;

const reg_t mode = cfg & 0x7;
const reg_t shift = (cfg >> 3) & 0x3f;
const reg_t count = (cfg >> 9) & 0xffffffff;

require((cfg >> 41) == 0);
require(mode <= 4);
require(count != 0);

if (mode == 4) {
  require(shift <= 31);
} else {
  require(shift != 0);
}

// Only mode 3 subtracts an input zero point.
require(mode == 3 || STATE.xrequant_input_zp == 0);

// Modes 0 and 3 have no output-zero-point addition.
require(
    (mode != 0 && mode != 3) ||
    STATE.xrequant_output_zp == 0
);

const int64_t multiplier = STATE.xrequant_multiplier;

if (mode == 4) {
  require(multiplier > 0);
  require(multiplier <= INT32_MAX);
}

const reg_t src_base = STATE.xrequant_src_base;
const reg_t dst_base = STATE.xrequant_dst_base;

const reg_t src_bytes =
    mode == 3 ? 1 :
    (mode == 2 || mode == 4) ? 4 : 8;

const bool output_is_int32 = mode == 0 || mode == 3;
const reg_t dst_bytes = output_is_int32 ? 4 : 1;

const int64_t output_zp =
    static_cast<int64_t>(STATE.xrequant_output_zp);

/*
 * Mode 4's first rounding is always Q31.
 * The conditional avoids evaluating shift-1 when its second shift is 0.
 */
const int64_t rounding_offset =
    mode == 4
        ? (INT64_C(1) << 30)
        : (INT64_C(1) << (shift - 1));

const int64_t second_mask =
    mode == 4
        ? ((INT64_C(1) << shift) - 1)
        : 0;

// Caller supplies adequate, non-overlapping buffers.
for (reg_t i = 0; i < count; ++i) {
  const reg_t src_address = src_base + i * src_bytes;
  const reg_t dst_address = dst_base + i * dst_bytes;

  int64_t input;

  if (mode == 3) {
    const int64_t raw =
        static_cast<int64_t>(MMU.load<int8_t>(src_address));

    const int64_t input_zp =
        static_cast<int64_t>(STATE.xrequant_input_zp);

    input = raw - input_zp;
  } else if (mode == 2 || mode == 4) {
    input = static_cast<int64_t>(
        MMU.load<int32_t>(src_address));
  } else {
    input = MMU.load<int64_t>(src_address);
  }

  // Mode 0 bypasses multiplication.
  int64_t product = input;

  if (mode != 0) {
    if (__builtin_mul_overflow(input, multiplier, &product)) {
      throw trap_xrequant_overflow(insn.bits());
    }
  }

  int64_t rounded_product;

  if (__builtin_add_overflow(
          product, rounding_offset, &rounded_product)) {
    throw trap_xrequant_overflow(insn.bits());
  }

  int64_t shifted;

  if (mode == 4) {
    // Stage 1: Q31 rounding, ties toward positive infinity.
    const int64_t high = rounded_product >> 31;

    // Stage 2: rounded division, ties away from zero.
    const int64_t remainder = high & second_mask;
    const int64_t threshold =
        (second_mask >> 1) + (high < 0);

    shifted =
        (high >> shift) + (remainder > threshold);
  } else {
    shifted = rounded_product >> shift;
  }

  if (output_is_int32) {
    if (shifted < INT32_MIN || shifted > INT32_MAX) {
      throw trap_xrequant_overflow(insn.bits());
    }

    MMU.store<uint32_t>(
        dst_address, static_cast<uint32_t>(shifted));

    continue;
  }

  // Modes 1, 2, and 4 produce saturated INT8 output.
  int64_t result;

  if (__builtin_add_overflow(shifted, output_zp, &result)) {
    throw trap_xrequant_overflow(insn.bits());
  }

  if (result < -128) {
    result = -128;
  } else if (result > 127) {
    result = 127;
  }

  MMU.store<uint8_t>(
      dst_address, static_cast<uint8_t>(result));
}

// Retain setup. Exceptions propagate; completed stores remain.

// REQUANT-ENGINE-MODIFICATION-END