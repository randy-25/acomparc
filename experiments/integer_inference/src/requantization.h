#ifndef INTEGER_INFERENCE_REQUANTIZATION_H
#define INTEGER_INFERENCE_REQUANTIZATION_H

#include <stddef.h>
#include <stdint.h>

/*
 * output[i] = (input[i] + 2^(shift-1)) >> shift
 * Rounding: nearest, ties toward positive infinity.
 *
 * Preconditions:
 * - shift is between 1 and 63.
 * - input[i] + rounding offset fits INT64.
 * - Rounded output fits INT32; no saturation is performed.
 * - Input/output buffers are valid and non-overlapping.
 *
 * Requires arithmetic signed right shift, as used by our GCC targets.
 */
void requantize_shift_i64_to_i32(
    const int64_t *input,
    int32_t *output,
    size_t element_count,
    unsigned int shift
);

/*
 * Multiplier-based conversion of centered INT64 values to INT8.
 * Rounding: nearest, ties toward positive infinity.
 *
 * Preconditions:
 * - shift is between 1 and 63.
 * - Multiplication and rounding-offset addition fit INT64.
 * - Input/output buffers are valid and non-overlapping.
 */
void requantize_i64_to_i8(
    const int64_t *input,
    int8_t *output,
    size_t element_count,
    int64_t multiplier,
    unsigned int shift,
    int32_t output_zero_point
);

/*
 * Convert centered INT32 accumulators to INT8.
 * Rounding: nearest, ties toward positive infinity.
 *
 * Preconditions:
 * - shift is between 1 and 63.
 * - Multiplication and rounding-offset addition fit INT64.
 * - Input/output buffers are valid and non-overlapping.
 *
 * Output zero point is added after rounding, before saturation.
 */
void requantize_i32_to_i8(
    const int32_t *input,
    int8_t *output,
    size_t element_count,
    int64_t multiplier,
    unsigned int shift,
    int32_t output_zero_point
);

/*
 * Convert centered INT8 values to INT32, output zero point 0.
 *
 * output[i] =
 *     ((input[i] - input_zero_point) * multiplier
 *      + 2^(shift-1)) >> shift
 *
 * shift must be in [1,63].
 * Centered values, products, and rounded results must fit their types.
 * Product + rounding offset must fit INT64.
 * Input/output buffers must not overlap.
 *
 * Requires arithmetic signed right shift, as provided by our GCC builds.
 */
void requantize_i8_to_i32(
    const int8_t *input,
    int32_t *output,
    size_t count,
    int64_t multiplier,
    unsigned int shift,
    int32_t input_zero_point
);

/*
 * Two-stage INT32 -> INT8 requantization.
 *
 * Stage 1:
 *   high = (input * q31_multiplier + 2^30) >> 31
 *   Nearest rounding, ties toward +infinity.
 *
 * Stage 2:
 *   Divide high by 2^second_right_shift.
 *   Nearest rounding, ties away from zero.
 *
 * Finally add output_zero_point and saturate to INT8.
 *
 * q31_multiplier must be positive and fit INT32.
 * second_right_shift must be in [0,31].
 * Input/output buffers must not overlap.
 *
 * Requires arithmetic signed right shift and two's-complement
 * signed integers, as provided by our GCC builds.
 */
void requantize_two_round_i32_to_i8(
    const int32_t *input,
    int8_t *output,
    size_t count,
    int32_t q31_multiplier,
    unsigned int second_right_shift,
    int32_t output_zero_point
);

#endif