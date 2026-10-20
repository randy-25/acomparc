#include "requantization.h"

void requantize_shift_i64_to_i32(
    const int64_t *input,
    int32_t *output,
    size_t element_count,
    unsigned int shift
)
{
    const int64_t rounding_offset =
        INT64_C(1) << (shift - 1);

    for (size_t i = 0; i < element_count; ++i) {
        const int64_t rounded =
            (input[i] + rounding_offset) >> shift;

        output[i] = (int32_t)rounded;
    }
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
    const int64_t rounding_offset =
        INT64_C(1) << (shift - 1);

    for (size_t i = 0; i < element_count; ++i) {
        const int64_t product = input[i] * multiplier;

        int64_t converted =
            (product + rounding_offset) >> shift;

        converted += (int64_t)output_zero_point;

        if (converted < -128) {
            converted = -128;
        } else if (converted > 127) {
            converted = 127;
        }

        output[i] = (int8_t)converted;
    }
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
    const int64_t rounding_offset =
        INT64_C(1) << (shift - 1);

    for (size_t i = 0; i < element_count; ++i) {
        /* Promote before multiplication. */
        const int64_t product =
            (int64_t)input[i] * multiplier;

        int64_t converted =
            (product + rounding_offset) >> shift;

        converted += (int64_t)output_zero_point;

        if (converted < -128) {
            converted = -128;
        } else if (converted > 127) {
            converted = 127;
        }

        output[i] = (int8_t)converted;
    }
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
    const int64_t rounding = INT64_C(1) << (shift - 1u);

    for (size_t i = 0; i < count; ++i) {
        const int64_t centered =
            (int64_t)input[i] - input_zero_point;

        output[i] = (int32_t)(
            (centered * multiplier + rounding) >> shift
        );
    }
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
    const int64_t mask =
        (INT64_C(1) << second_right_shift) - 1;

    for (size_t i = 0; i < count; ++i) {
        /* Stage 1: rounded Q31 multiplication. */
        const int64_t product =
            (int64_t)input[i] * q31_multiplier;

        const int64_t high =
            (product + (INT64_C(1) << 30)) >> 31;

        /* Stage 2: rounded division, ties away from zero. */
        const int64_t remainder = high & mask;

        const int64_t threshold =
            (mask >> 1) + (high < 0);

        const int64_t rounded =
            (high >> second_right_shift)
            + (remainder > threshold);

        int64_t result = rounded + output_zero_point;

        if (result < -128) {
            result = -128;
        } else if (result > 127) {
            result = 127;
        }

        output[i] = (int8_t)result;
    }
}