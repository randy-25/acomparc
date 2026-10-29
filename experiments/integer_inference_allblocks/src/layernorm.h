#ifndef TINYSTORIES_ALLBLOCKS_LAYERNORM_H
#define TINYSTORIES_ALLBLOCKS_LAYERNORM_H

#include <stddef.h>
#include <stdint.h>
#include "integer_parameter_types.h"

/* LayerNormI8Params comes from the generated parameter type header.
 * All input/output/workspace buffers must be valid and mutually disjoint.
 * Sizes for a complete LN1/LN2: 32 for each row buffer; 4096 for each feature
 * buffer. Every intermediate must satisfy the bounds checked in Python.
 */
typedef struct {
    int32_t *mean_sum;
    int32_t *variance;
    int32_t *inv_std;
    int32_t *feature_i32;
    int64_t *fine_i64;
} LayerNormI8Workspace;

/* Kernels perform arithmetic only; conversions are separate calls.
 * A feature sum represents a mean at input_scale / feature_count.
 * Squared difference: (input * feature_count - mean_sum)^2.
 * Normalize: (input * feature_count - mean_sum) * inv_std.
 * Affine: normalized * gamma + beta * beta_alignment.
 * Integer reciprocal square root: floor(sqrt(floor(numerator / input))).
 * Counts must be positive; denominators must be positive. Integer sums,
 * products, squares and narrowing results must fit their destination types.
 * add_epsilon and rsqrt permit identical input/output buffers.
 */
void layernorm_sum_i8(const int8_t *, int32_t *, size_t, size_t);
void layernorm_squared_difference_i8(const int8_t *, const int32_t *,
                                     int64_t *, size_t, size_t);
void layernorm_variance_i32(const int32_t *, int32_t *, size_t, size_t);
void layernorm_add_epsilon_i32(const int32_t *, int32_t *, size_t, int32_t);
void layernorm_rsqrt_i32(const int32_t *, int32_t *, size_t, uint64_t);
void layernorm_normalize_i8(const int8_t *, const int32_t *, const int32_t *,
                            int64_t *, size_t, size_t);
void layernorm_affine_i32(const int32_t *, const int32_t *, const int32_t *,
                          int64_t *, size_t, size_t, int64_t);

/* INT32-input variants retained for Part 5 (LN2). Input residual scale 2^-16. */
void layernorm_sum_i32(const int32_t *, int32_t *, size_t, size_t);
void layernorm_squared_difference_i32(const int32_t *, const int32_t *,
                                      int64_t *, size_t, size_t);
void layernorm_normalize_i32(const int32_t *, const int32_t *, const int32_t *,
                             int64_t *, size_t, size_t);
#endif
