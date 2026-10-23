#ifndef INTEGER_INFERENCE_RESIDUAL_H
#define INTEGER_INFERENCE_RESIDUAL_H

#include <stddef.h>
#include <stdint.h>

/*
 * Add an INT8 residual input to an already aligned INT32 branch.
 *
 * output[i] = branch[i]
 *           + (residual[i] - residual_zero_point) * alignment
 *
 * alignment converts the residual input exactly to the branch scale.
 * Branch and output have zero point 0.
 * Intermediate calculations must fit INT64; outputs must fit INT32.
 *
 * output may equal branch; otherwise, they must not overlap.
 * The residual buffer must not overlap branch or output.
 * No rounding or saturation is performed.
 */
void residual_add_i8_i32(
    const int8_t *residual,
    const int32_t *branch,
    int32_t *output,
    size_t count,
    int64_t alignment,
    int32_t residual_zero_point
);

/*
 * Add two INT32 tensors with identical scale and zero point 0.
 *
 * output[i] = left[i] + right[i]
 *
 * Results must fit INT32.
 * Output may equal either input; otherwise, it must not overlap them.
 * No requantization or saturation is performed.
 */
void residual_add_i32(
    const int32_t *left,
    const int32_t *right,
    int32_t *output,
    size_t count
);

#endif