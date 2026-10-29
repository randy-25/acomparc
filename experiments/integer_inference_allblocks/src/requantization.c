#include "requantization.h"
#include "xrequant.h"

static inline void requantize_buffer(
    unsigned int mode,
    const void *input,
    void *output,
    size_t count,
    int64_t multiplier,
    unsigned int shift,
    int32_t input_zero_point,
    int32_t output_zero_point
)
{
    if (count == 0) {
        return;
    }

    /*
     * Validate before packing:
     * MODE:  3 bits, currently 0..4.
     * SHIFT: 6 bits, with mode-specific limits.
     * COUNT: 32 bits, stored directly.
     */
    if (mode > 4 ||
        count > UINT32_MAX ||
        (mode == 4
            ? shift > 31
            : (shift == 0 || shift > 63))) {
        __builtin_trap();
    }

    const uint64_t cfg =
        (uint64_t)mode |
        ((uint64_t)shift << 3) |
        ((uint64_t)count << 9);

    xrequant_cfg(cfg);
    xrequant_set_io(input, output);
    xrequant_set_mul(multiplier);
    xrequant_set_zp(input_zero_point, output_zero_point);
    xrequant_run();
}

void requantize_shift_i64_to_i32(
    const int64_t *input,
    int32_t *output,
    size_t element_count,
    unsigned int shift
)
{
    /*
     * Mode 0: rounded shift, no multiplication or saturation.
     * SET_MUL remains required; its value is ignored by this mode.
     */
    requantize_buffer(
        0,
        input,
        output,
        element_count,
        INT64_C(1),
        shift,
        0,
        0
    );
}

void requantize_i64_to_i8(
    const int64_t *input,
    int8_t *output,
    size_t element_count,
    int64_t multiplier,
    unsigned int shift,
    int32_t output_zero_point
)
{
    requantize_buffer(
        1,
        input,
        output,
        element_count,
        multiplier,
        shift,
        0,
        output_zero_point
    );
}

void requantize_i32_to_i8(
    const int32_t *input,
    int8_t *output,
    size_t element_count,
    int64_t multiplier,
    unsigned int shift,
    int32_t output_zero_point
)
{
    requantize_buffer(
        2,
        input,
        output,
        element_count,
        multiplier,
        shift,
        0,
        output_zero_point
    );
}

void requantize_i8_to_i32(
    const int8_t *input,
    int32_t *output,
    size_t count,
    int64_t multiplier,
    unsigned int shift,
    int32_t input_zero_point
)
{
    requantize_buffer(
        3,
        input,
        output,
        count,
        multiplier,
        shift,
        input_zero_point,
        0
    );
}

void requantize_two_round_i32_to_i8(
    const int32_t *input,
    int8_t *output,
    size_t count,
    int32_t q31_multiplier,
    unsigned int second_right_shift,
    int32_t output_zero_point
)
{
    /*
     * Mode 4 fixes the first shift at 31.
     * CFG's SHIFT field supplies the second shift.
     */
    requantize_buffer(
        4,
        input,
        output,
        count,
        (int64_t)q31_multiplier,
        second_right_shift,
        0,
        output_zero_point
    );
}