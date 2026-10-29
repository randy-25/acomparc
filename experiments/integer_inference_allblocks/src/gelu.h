#ifndef INTEGER_INFERENCE_GELU_H
#define INTEGER_INFERENCE_GELU_H

#include <stddef.h>
#include <stdint.h>

/*
 * Multiply centered INT8 input by a centered integer scalar.
 *
 * output[i] = (input[i] - input_zero_point) * centered_scalar
 *
 * Output: INT32, zero point 0.
 * Output scale = input scale * scalar scale.
 *
 * Intermediate products must fit INT64; outputs must fit INT32.
 * Caller supplies valid, non-overlapping input/output buffers.
 * No requantization is performed.
 */
void gelu_multiply_scalar_i8(
    const int8_t *input,
    int32_t *output,
    size_t count,
    int32_t input_zero_point,
    int32_t centered_scalar
);

/*
 * Signed cube of centered INT8 input.
 *
 * output[i] = (input[i] - input_zero_point)^3
 *
 * Input zero point must be in [-128,127].
 * Output: INT32, scale = input_scale^3, zero point 0.
 *
 * The full centered INT8 range is at most [-255,255].
 * Its square and cube fit INT32.
 *
 * Input/output buffers must not overlap.
 * No requantization is performed.
 */
void gelu_cube_i8(
    const int8_t *input,
    int32_t *output,
    size_t count,
    int32_t input_zero_point
);

/*
 * Combine two centered INT8 branches using fixed-point multipliers.
 *
 * output[i] =
 *     (linear[i] - linear_zero_point) * linear_multiplier
 *   + (cubic[i]  - cubic_zero_point)  * cubic_multiplier
 *
 * The multipliers share a common fractional-bit count.
 * Output: INT64 combined product, zero point 0.
 *
 * Each product and their sum must fit INT64.
 * Output must not overlap either input.
 * No rounding, requantization, or saturation is performed.
 */
void gelu_add_scaled_i8(
    const int8_t *linear,
    const int8_t *cubic,
    int64_t *output,
    size_t count,
    int32_t linear_zero_point,
    int32_t cubic_zero_point,
    int64_t linear_multiplier,
    int64_t cubic_multiplier
);

/*
 * Direct tanh lookup for an INT8 input code.
 *
 * table[input_code + 128] supplies the output.
 * The 256-entry table encodes the input scale and zero point.
 *
 * Block-0 output: INT32, scale 2^-24, zero point 0.
 * Input and table must not overlap the output buffer.
 */
void gelu_tanh_lut_i8(
    const int8_t *input,
    int32_t *output,
    size_t count,
    const int32_t *table
);

/*
 * Add an integer scalar at the input's scale.
 *
 * output[i] = input[i] + scalar
 *
 * Results must fit INT32.
 * Input/output may be identical; otherwise, they must not overlap.
 * No requantization is performed.
 */
void gelu_add_scalar_i32(
    const int32_t *input,
    int32_t *output,
    size_t count,
    int32_t scalar
);

/*
 * Elementwise product of centered INT8 inputs.
 *
 * output[i] = (left[i] - left_zero_point)
 *           * (right[i] - right_zero_point)
 *
 * Both zero points must be in [-128,127].
 * Products fit INT32 over the full INT8 input domain.
 * Output scale = left scale * right scale; output zero point 0.
 *
 * Output must not overlap either input.
 * No requantization is performed.
 */
void gelu_multiply_i8(
    const int8_t *left,
    const int8_t *right,
    int32_t *output,
    size_t count,
    int32_t left_zero_point,
    int32_t right_zero_point
);

#endif