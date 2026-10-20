#ifndef INTEGER_INFERENCE_LAYERNORM_H
#define INTEGER_INFERENCE_LAYERNORM_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const int32_t *gamma;
    const int32_t *beta;

    unsigned int square_shift;
    int32_t epsilon_q;
    uint64_t rsqrt_numerator;
    unsigned int normalize_shift;
    int64_t beta_alignment;

    int64_t output_multiplier;
    unsigned int output_shift;
    int32_t output_zero_point;
} LayerNormI8Params;

typedef struct {
    int32_t *mean_sum;
    int32_t *variance;
    int32_t *inv_std;

    int32_t *feature_i32;
    int64_t *fine_i64;
} LayerNormI8Workspace;

/*
 * Input: token-major INT8 [token_count, feature_count], zero point 0.
 * Output: one INT32 feature sum per token.
 *
 * Caller supplies valid, non-overlapping input/output buffers.
 * Each feature sum must fit INT32.
 * For our case: 32 tokens, 128 features.
 *
 * Output scale = input scale / feature_count when interpreted as a mean.
 * No division or requantization is performed.
 */
void layernorm_sum_i8(
    const int8_t *input,
    int32_t *output,
    size_t token_count,
    size_t feature_count
);

/*
 * Current contract:
 * Input: INT8 [32,128], scale 2^-8, zero point 0.
 * Mean sums: INT32 [32], scale 2^-15, zero point 0.
 * Output: INT64 [32,128], scale 2^-30, zero point 0.
 *
 * Caller supplies valid, non-overlapping buffers.
 * No requantization is performed.
 */
void layernorm_squared_difference_i8(
    const int8_t *input,
    const int32_t *mean_sum,
    int64_t *output,
    size_t token_count,
    size_t feature_count
);

/*
 * Sum squared differences across features.
 *
 * Block-0 LN1:
 * Input: INT32 [32,128], scale 2^-15.
 * Output: INT32 [32], scale 2^-22.
 * Zero points: 0.
 *
 * Caller supplies valid, non-overlapping buffers.
 * Accumulation must fit INT64; final sum must fit INT32.
 * No division or requantization is performed.
 */
void layernorm_variance_i32(
    const int32_t *squared_difference,
    int32_t *output,
    size_t token_count,
    size_t feature_count
);

/*
 * Op 4: add epsilon encoded at the variance scale.
 * Input/output: INT32, same scale, zero point 0.
 *
 * Variance must be nonnegative; epsilon_q must be positive.
 * The sum must fit INT32.
 * Input and output may be the same buffer.
 */
void layernorm_add_epsilon_i32(
    const int32_t *variance,
    int32_t *output,
    size_t token_count,
    int32_t epsilon_q
);

/*
 * Op 5: output[t] = isqrt(numerator / input[t]).
 *
 * Input values must be positive.
 * Integer division and square root use floor rounding.
 * Results must fit INT32.
 *
 * Block-0 LN1: numerator = 2^54.
 * Input scale: 2^-22; output scale: 2^-16.
 * Input and output may be the same buffer.
 */
void layernorm_rsqrt_i32(
    const int32_t *input,
    int32_t *output,
    size_t token_count,
    uint64_t numerator
);


/*
 * Op 6–11: normalized features before requantization.
 *
 * Block-0 LN1:
 * input:    INT8  [32,128], scale 2^-8.
 * mean_sum: INT32 [32],     scale 2^-15.
 * inv_std:  INT32 [32],     scale 2^-16.
 * output:   INT64 [32,128], scale 2^-31.
 * All zero points are 0.
 *
 * Caller supplies valid, non-overlapping buffers.
 * Intermediate calculations must fit INT64.
 * No requantization is performed.
 */
void layernorm_normalize_i8(
    const int8_t *input,
    const int32_t *mean_sum,
    const int32_t *inv_std,
    int64_t *output,
    size_t token_count,
    size_t feature_count
);

/*
 * Op 12–13: affine transform.
 *
 * Block-0 LN1:
 * normalized: INT32 [32,128], scale 2^-16.
 * gamma/beta: INT32 [128],    scale 2^-16.
 * output:     INT64 [32,128], scale 2^-32.
 * All zero points are 0.
 *
 * beta_alignment converts beta exactly to the product scale.
 * Caller supplies valid, non-overlapping buffers.
 * Products and their sum must fit INT64.
 * No rounding or requantization is performed.
 */
void layernorm_affine_i32(
    const int32_t *normalized,
    const int32_t *gamma,
    const int32_t *beta,
    int64_t *output,
    size_t token_count,
    size_t feature_count,
    int64_t beta_alignment
);


/*
 * Sum INT32 features for each token.
 *
 * Input: token-major [token_count, feature_count], zero point 0.
 * Output: INT32 [token_count].
 *
 * When interpreted as a mean:
 *     output scale = input scale / feature_count.
 *
 * LN2: input scale 2^-16; mean-sum scale 2^-23.
 *
 * Every partial sum must fit INT64; final sums must fit INT32.
 * Input/output buffers must not overlap.
 * No division or requantization is performed.
 */
void layernorm_sum_i32(
    const int32_t *input,
    int32_t *output,
    size_t token_count,
    size_t feature_count
);

/*
 * Calculate squared differences using stored mean sums:
 *
 * difference = input[t,k] * feature_count - mean_sum[t]
 * output[t,k] = difference * difference
 *
 * LN2:
 * input:    INT32 [32,128], scale 2^-16.
 * mean_sum: INT32 [32],     scale 2^-23.
 * output:   INT64 [32,128], scale 2^-46.
 * All zero points are 0.
 *
 * feature_count must be positive and fit INT64.
 * Alignment, subtraction, and squares must fit INT64.
 * Caller supplies valid, non-overlapping buffers.
 * No requantization is performed.
 */
void layernorm_squared_difference_i32(
    const int32_t *input,
    const int32_t *mean_sum,
    int64_t *output,
    size_t token_count,
    size_t feature_count
);

/*
 * Calculate normalized features before requantization:
 *
 * difference = input[t,k] * feature_count - mean_sum[t]
 * output[t,k] = difference * inv_std[t]
 *
 * LN2:
 * input:    INT32 [32,128], scale 2^-16.
 * mean_sum: INT32 [32],     scale 2^-23.
 * inv_std:  INT32 [32],     scale 2^-16.
 * output:   INT64 [32,128], scale 2^-39.
 * All zero points are 0.
 *
 * feature_count must be positive and fit INT64.
 * Alignment, subtraction, and products must fit INT64.
 * Caller supplies valid, non-overlapping buffers.
 * No requantization is performed.
 */
void layernorm_normalize_i32(
    const int32_t *input,
    const int32_t *mean_sum,
    const int32_t *inv_std,
    int64_t *output,
    size_t token_count,
    size_t feature_count
);

#endif