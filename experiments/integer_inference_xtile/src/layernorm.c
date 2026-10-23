#include "layernorm.h"

void layernorm_sum_i8(
    const int8_t *input,
    int32_t *output,
    size_t token_count,
    size_t feature_count
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const int8_t *row = input + t * feature_count;
        int32_t sum = 0;

        for (size_t k = 0; k < feature_count; ++k) {
            sum += (int32_t)row[k];
        }

        output[t] = sum;
    }
}

void layernorm_squared_difference_i8(
    const int8_t *input,
    const int32_t *mean_sum,
    int64_t *output,
    size_t token_count,
    size_t feature_count
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const size_t offset = t * feature_count;
        const int64_t mean = (int64_t)mean_sum[t];

        for (size_t k = 0; k < feature_count; ++k) {
            const int64_t aligned_input =
                (int64_t)input[offset + k]
                * (int64_t)feature_count;

            const int64_t difference = aligned_input - mean;

            output[offset + k] = difference * difference;
        }
    }
}


void layernorm_variance_i32(
    const int32_t *squared_difference,
    int32_t *output,
    size_t token_count,
    size_t feature_count
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const int32_t *row =
            squared_difference + t * feature_count;

        int64_t sum = 0;

        for (size_t k = 0; k < feature_count; ++k) {
            sum += (int64_t)row[k];
        }

        output[t] = (int32_t)sum;
    }
}

void layernorm_add_epsilon_i32(
    const int32_t *variance,
    int32_t *output,
    size_t token_count,
    int32_t epsilon_q
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const int64_t sum =
            (int64_t)variance[t] + (int64_t)epsilon_q;

        output[t] = (int32_t)sum;
    }
}

/* Exact floor square root using unsigned integer arithmetic. */
static uint64_t isqrt_u64(uint64_t value)
{
    uint64_t root = 0;
    uint64_t bit = UINT64_C(1) << 62;

    /* Start at the largest power of four not exceeding value. */
    while (bit > value) {
        bit >>= 2;
    }

    while (bit != 0) {
        if (value >= root + bit) {
            value -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }

        bit >>= 2;
    }

    return root;
}

void layernorm_rsqrt_i32(
    const int32_t *input,
    int32_t *output,
    size_t token_count,
    uint64_t numerator
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const uint64_t quotient =
            numerator / (uint64_t)input[t];

        output[t] = (int32_t)isqrt_u64(quotient);
    }
}

void layernorm_normalize_i8(
    const int8_t *input,
    const int32_t *mean_sum,
    const int32_t *inv_std,
    int64_t *output,
    size_t token_count,
    size_t feature_count
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const size_t offset = t * feature_count;
        const int64_t mean = (int64_t)mean_sum[t];
        const int64_t inverse_std = (int64_t)inv_std[t];

        for (size_t k = 0; k < feature_count; ++k) {
            const int64_t aligned_input =
                (int64_t)input[offset + k]
                * (int64_t)feature_count;

            const int64_t difference = aligned_input - mean;

            output[offset + k] = difference * inverse_std;
        }
    }
}

void layernorm_affine_i32(
    const int32_t *normalized,
    const int32_t *gamma,
    const int32_t *beta,
    int64_t *output,
    size_t token_count,
    size_t feature_count,
    int64_t beta_alignment
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const size_t offset = t * feature_count;

        for (size_t k = 0; k < feature_count; ++k) {
            const int64_t product =
                (int64_t)normalized[offset + k]
                * (int64_t)gamma[k];

            const int64_t aligned_beta =
                (int64_t)beta[k] * beta_alignment;

            output[offset + k] = product + aligned_beta;
        }
    }
}

void layernorm_sum_i32(
    const int32_t *input,
    int32_t *output,
    size_t token_count,
    size_t feature_count
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const int32_t *input_row = input + t * feature_count;
        int64_t sum = 0;

        for (size_t k = 0; k < feature_count; ++k) {
            sum += (int64_t)input_row[k];
        }

        output[t] = (int32_t)sum;
    }
}


void layernorm_squared_difference_i32(
    const int32_t *input,
    const int32_t *mean_sum,
    int64_t *output,
    size_t token_count,
    size_t feature_count
)
{
    const int64_t alignment = (int64_t)feature_count;

    for (size_t t = 0; t < token_count; ++t) {
        const int32_t *input_row = input + t * feature_count;
        int64_t *output_row = output + t * feature_count;

        const int64_t mean = mean_sum[t];

        for (size_t k = 0; k < feature_count; ++k) {
            const int64_t aligned_input =
                (int64_t)input_row[k] * alignment;

            const int64_t difference = aligned_input - mean;

            output_row[k] = difference * difference;
        }
    }
}


void layernorm_normalize_i32(
    const int32_t *input,
    const int32_t *mean_sum,
    const int32_t *inv_std,
    int64_t *output,
    size_t token_count,
    size_t feature_count
)
{
    const int64_t alignment = (int64_t)feature_count;

    for (size_t t = 0; t < token_count; ++t) {
        const int32_t *input_row = input + t * feature_count;
        int64_t *output_row = output + t * feature_count;

        const int64_t mean = mean_sum[t];
        const int64_t inverse_std = inv_std[t];

        for (size_t k = 0; k < feature_count; ++k) {
            const int64_t aligned_input =
                (int64_t)input_row[k] * alignment;

            const int64_t difference = aligned_input - mean;

            output_row[k] = difference * inverse_std;
        }
    }
}