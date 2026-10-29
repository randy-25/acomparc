#ifndef INTEGER_INFERENCE_PROJECTION_H
#define INTEGER_INFERENCE_PROJECTION_H

#include <stddef.h>
#include <stdint.h>

/*
 * INT8 fully connected accumulation.
 *
 * input:   [token_count, input_features], token-major.
 * weights: [output_features, input_features], row-major.
 * bias:    [output_features], or NULL when absent.
 * output:  INT32 [token_count, output_features].
 *
 * Weight zero point must be 0.
 * Input zero point must be within the INT8 range.
 * Bias scale must equal input_scale * weight_scale.
 * Output accumulator scale = input_scale * weight_scale.
 *
 * Every product, partial sum, and bias addition must fit INT32.
 * Caller supplies valid, non-overlapping buffers.
 * No requantization or saturation is performed.
 */
void projection_accumulate_i8(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    size_t token_count,
    size_t input_features,
    size_t output_features,
    int32_t input_zero_point
);

#endif