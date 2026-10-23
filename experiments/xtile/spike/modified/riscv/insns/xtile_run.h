// TILE-ENGINE-MODIFICATION-START

require_rv64;

// All setup instructions must have completed since reset.
require(STATE.xtile_cfg_valid);
require(STATE.xtile_ab_valid);
require(STATE.xtile_c_valid);
require(STATE.xtile_zp_valid);

const reg_t cfg = STATE.xtile_cfg;

// Operand types: false = signed INT8, true = signed INT32.
const bool a_is_int32 = (cfg & 0x1) != 0;
const bool b_is_int32 = (cfg & 0x2) != 0;

// Output width is derived from the operand types.
const bool wide_output = a_is_int32 || b_is_int32;

const reg_t a_bytes = a_is_int32 ? 4 : 1;
const reg_t b_bytes = b_is_int32 ? 4 : 1;
const reg_t c_bytes = wide_output ? 8 : 4;

// Active tile dimensions are encoded minus one.
const reg_t tile_rows = ((cfg >> 2) & 0x7) + 1;
const reg_t tile_columns = ((cfg >> 5) & 0x7) + 1;

// Matrix dimensions were validated by xtile.cfg.
const reg_t m_count = (cfg >> 8) & 0x3ff;
const reg_t n_count = (cfg >> 18) & 0x3ff;
const reg_t k_count = (cfg >> 28) & 0x3ff;

const reg_t a_base = STATE.xtile_a_base;
const reg_t bt_base = STATE.xtile_bt_base;
const reg_t c_base = STATE.xtile_c_base;

const int64_t a_zero_point =
    static_cast<int64_t>(STATE.xtile_a_zp);

const int64_t b_zero_point =
    static_cast<int64_t>(STATE.xtile_b_zp);

// Validate type-dependent zero-point ranges before memory access.
require(a_is_int32 ||
        (a_zero_point >= -128 && a_zero_point <= 127));

require(b_is_int32 ||
        (b_zero_point >= -128 && b_zero_point <= 127));

// Software must provide adequate buffers and keep C separate
// from A and BT. Input matrices may share storage with each other.

// Visit output tiles in increasing row, then column order.
for (reg_t m_start = 0;
     m_start < m_count;
     m_start += tile_rows) {

  const reg_t rows_remaining = m_count - m_start;
  const reg_t active_rows =
      rows_remaining < tile_rows ? rows_remaining : tile_rows;

  for (reg_t n_start = 0;
       n_start < n_count;
       n_start += tile_columns) {

    const reg_t columns_remaining = n_count - n_start;
    const reg_t active_columns =
        columns_remaining < tile_columns
            ? columns_remaining
            : tile_columns;

    // Scratch belongs to the host simulator.
    // It models one output tile, not the entire C matrix.
    int64_t accumulators[8][8] = {};

    // Accumulate each output in increasing K order.
    for (reg_t k = 0; k < k_count; ++k) {

      int64_t a_operands[8];
      int64_t b_operands[8];

      // Load one centered A operand for each active row.
      for (reg_t row = 0; row < active_rows; ++row) {

        const reg_t m = m_start + row;

        const reg_t address =
            a_base + (m * k_count + k) * a_bytes;

        const int64_t value =
            a_is_int32
                ? static_cast<int64_t>(
                      MMU.load<int32_t>(address))
                : static_cast<int64_t>(
                      MMU.load<int8_t>(address));

        a_operands[row] = value - a_zero_point;
      }

      // Load one centered BT operand for each active column.
      for (reg_t column = 0;
           column < active_columns;
           ++column) {

        const reg_t n = n_start + column;

        const reg_t address =
            bt_base + (n * k_count + k) * b_bytes;

        const int64_t value =
            b_is_int32
                ? static_cast<int64_t>(
                      MMU.load<int32_t>(address))
                : static_cast<int64_t>(
                      MMU.load<int8_t>(address));

        b_operands[column] = value - b_zero_point;
      }

      // Update all active output accumulators.
      for (reg_t row = 0; row < active_rows; ++row) {

        for (reg_t column = 0;
             column < active_columns;
             ++column) {

          int64_t product;

          if (__builtin_mul_overflow(
                  a_operands[row],
                  b_operands[column],
                  &product)) {
            throw trap_xtile_overflow(insn.bits());
          }

          int64_t next;

          if (__builtin_add_overflow(
                  accumulators[row][column],
                  product,
                  &next)) {
            throw trap_xtile_overflow(insn.bits());
          }

          accumulators[row][column] = next;
        }
      }
    }

    // All arithmetic for this tile succeeded.
    // Store valid elements in row-major order.
    for (reg_t row = 0; row < active_rows; ++row) {

      const reg_t m = m_start + row;

      for (reg_t column = 0;
           column < active_columns;
           ++column) {

        const reg_t n = n_start + column;

        const reg_t address =
            c_base + (m * n_count + n) * c_bytes;

        const int64_t result = accumulators[row][column];

        if (wide_output) {
          MMU.store<uint64_t>(
              address, static_cast<uint64_t>(result));
        } else {
          // INT8/INT8 sums fit INT32 under the validated
          // zero-point ranges and configured K bound.
          MMU.store<uint32_t>(
              address, static_cast<uint32_t>(result));
        }
      }
    }
  }
}

// Retain configuration, addresses, zero points, and valid flags.
// MMU exceptions propagate normally; completed stores remain.

// TILE-ENGINE-MODIFICATION-END