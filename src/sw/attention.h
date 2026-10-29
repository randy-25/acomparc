#ifndef INTEGER_INFERENCE_ATTENTION_H
#define INTEGER_INFERENCE_ATTENTION_H

#include <stddef.h>
#include <stdint.h>

/*
 * Compute attention scores directly from token-major Q/K buffers.
 *
 * query/key:
 *   INT8 [token_count, head_count * head_dim].
 *
 * scores:
 *   INT32 [head_count, token_count, token_count].
 *   Axis order: [head, query_token, key_token].
 *   Scale = query_scale * key_scale; zero point = 0.
 *
 * Preconditions:
 * - Query/key zero points are within the INT8 range.
 * - Products and partial sums fit INT32.
 * - Buffer sizes and index calculations fit size_t.
 * - Input/output buffers are valid and non-overlapping.
 *
 * Calculates every score, including entries later masked out.
 * No masking, score scaling, or requantization is performed.
 */
void attention_scores_i8(
    const int8_t *query,
    const int8_t *key,
    int32_t *scores,
    size_t token_count,
    size_t head_count,
    size_t head_dim,
    int32_t query_zero_point,
    int32_t key_zero_point
);


/*
 * Causal masking and maximum subtraction.
 *
 * Input/output layout: [head_count, token_count, token_count].
 * Query t may attend only to keys j <= t.
 *
 * Allowed output:
 *     scores[h,t,j] - maximum_allowed_score[h,t]
 *
 * Masked output: placeholder 0; downstream softmax MUST ignore it.
 *
 * Output retains the input scale and has zero point 0.
 * Each allowed difference must fit INT32.
 *
 * token_count must be positive.
 * Buffers may be identical; otherwise, they must not overlap.
 */
void attention_causal_shift_i32(
    const int32_t *scores,
    int32_t *shifted,
    size_t head_count,
    size_t token_count
);


/*
 * LUT exponential for causally masked, maximum-subtracted scores.
 *
 * Input/output layout: [head_count, token_count, token_count].
 * Allowed inputs j <= t must be nonpositive.
 * Masked outputs j > t are exactly zero.
 *
 * LUT contract:
 *   4097 INT32 entries representing exp(-i / 256), scale 2^-30.
 *   Coordinate scale: 2^-16.
 *   Coordinates >= 16 * 65536 produce zero.
 *
 * Output: nonnegative INT32, scale 2^-30, zero point 0.
 *
 * coordinate_multiplier must be positive.
 * coordinate_shift must be in [1,63].
 * Magnitude * multiplier + rounding offset must fit INT64.
 *
 * Input/output may be identical; otherwise, they must not overlap.
 * The LUT must not overlap the output.
 *
 * Requires arithmetic right shift of negative signed integers,
 * as provided by our GCC host and RISC-V builds.
 */
void attention_exp_lut_i32(
    const int32_t *shifted,
    int32_t *exponentials,
    size_t head_count,
    size_t token_count,
    const int32_t *exp_lut,
    int64_t coordinate_multiplier,
    unsigned int coordinate_shift
);

/*
 * Normalize causal exponential rows into probabilities.
 *
 * Input/output layout: [head_count, token_count, token_count].
 * Only keys j <= t participate in query t's normalization.
 * Masked outputs j > t are exactly zero.
 *
 * Inputs must be nonnegative, with a positive allowed sum per row.
 * All input exponentials must share the same scale; it cancels
 * during division.
 *
 * Output: INT32, scale 2^-probability_frac_bits, zero point 0.
 * probability_frac_bits must be in [0,30].
 *
 * Row sums and numerator + rounding offset must fit INT64.
 * Input/output may be identical; otherwise, they must not overlap.
 */
void attention_normalize_exp_i32(
    const int32_t *exponentials,
    int32_t *probabilities,
    size_t head_count,
    size_t token_count,
    unsigned int probability_frac_bits
);


/*
 * Probability × V, with token-major context output.
 *
 * probabilities: INT32 [head_count, token_count, token_count].
 * value:         INT8  [token_count, head_count * head_dim].
 * context:       INT32 [token_count, head_count * head_dim].
 *
 * Probabilities have zero point 0.
 * value_zero_point is subtracted after promotion.
 *
 * Output scale = probability scale * value scale.
 * Output zero point = 0.
 *
 * Caller supplies valid, non-overlapping buffers.
 * Every product and partial sum must fit INT32.
 * No requantization is performed.
 */
void attention_context_i8(
    const int32_t *probabilities,
    const int8_t *value,
    int32_t *context,
    size_t token_count,
    size_t head_count,
    size_t head_dim,
    int32_t value_zero_point
);
#endif