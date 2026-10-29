#include "attention.h"

void attention_scores_i8(
    const int8_t *query,
    const int8_t *key,
    int32_t *scores,
    size_t token_count,
    size_t head_count,
    size_t head_dim,
    int32_t query_zero_point,
    int32_t key_zero_point
)
{
    const size_t hidden_size = head_count * head_dim;

    for (size_t h = 0; h < head_count; ++h) {
        const size_t head_offset = h * head_dim;

        for (size_t t = 0; t < token_count; ++t) {
            const int8_t *query_row =
                query + t * hidden_size + head_offset;

            int32_t *score_row =
                scores + (h * token_count + t) * token_count;

            for (size_t j = 0; j < token_count; ++j) {
                const int8_t *key_row =
                    key + j * hidden_size + head_offset;

                int32_t accumulator = 0;

                for (size_t d = 0; d < head_dim; ++d) {
                    const int32_t q_centered =
                        (int32_t)query_row[d] - query_zero_point;

                    const int32_t k_centered =
                        (int32_t)key_row[d] - key_zero_point;

                    accumulator += q_centered * k_centered;
                }

                score_row[j] = accumulator;
            }
        }
    }
}

void attention_causal_shift_i32(
    const int32_t *scores,
    int32_t *shifted,
    size_t head_count,
    size_t token_count
)
{
    for (size_t h = 0; h < head_count; ++h) {
        for (size_t t = 0; t < token_count; ++t) {
            const size_t row_offset =
                (h * token_count + t) * token_count;

            const int32_t *score_row = scores + row_offset;
            int32_t *shifted_row = shifted + row_offset;

            /* Only allowed keys participate in the maximum. */
            int32_t maximum = score_row[0];

            for (size_t j = 1; j <= t; ++j) {
                if (score_row[j] > maximum) {
                    maximum = score_row[j];
                }
            }

            /* Promote before subtraction; contract requires INT32 fit. */
            for (size_t j = 0; j <= t; ++j) {
                const int64_t difference =
                    (int64_t)score_row[j] - (int64_t)maximum;

                shifted_row[j] = (int32_t)difference;
            }

            /* These placeholders must be ignored by softmax. */
            for (size_t j = t + 1; j < token_count; ++j) {
                shifted_row[j] = 0;
            }
        }
    }
}

void attention_exp_lut_i32(
    const int32_t *shifted,
    int32_t *exponentials,
    size_t head_count,
    size_t token_count,
    const int32_t *exp_lut,
    int64_t coordinate_multiplier,
    unsigned int coordinate_shift
)
{
    const int64_t coordinate_rounding =
        INT64_C(1) << (coordinate_shift - 1u);

    /* 16 expressed at coordinate scale 2^-16. */
    const int64_t cutoff = INT64_C(16) << 16;

    for (size_t h = 0; h < head_count; ++h) {
        for (size_t t = 0; t < token_count; ++t) {
            const size_t row_offset =
                (h * token_count + t) * token_count;

            const int32_t *input_row = shifted + row_offset;
            int32_t *output_row = exponentials + row_offset;

            for (size_t j = 0; j <= t; ++j) {
                /* Promote before negation, including INT32_MIN. */
                const int64_t magnitude =
                    -(int64_t)input_row[j];

                const int64_t coordinate =
                    (magnitude * coordinate_multiplier
                     + coordinate_rounding)
                    >> coordinate_shift;

                if (coordinate >= cutoff) {
                    output_row[j] = 0;
                    continue;
                }

                /*
                 * Coordinate has 16 fractional bits.
                 * LUT spacing is 1/256: eight interpolation bits.
                 */
                const size_t index =
                    (size_t)(coordinate >> 8);

                const int64_t fraction =
                    coordinate & INT64_C(255);

                const int64_t lower = exp_lut[index];
                const int64_t upper = exp_lut[index + 1u];

                /* Nearest rounding, ties toward +infinity. */
                const int64_t interpolated =
                    lower
                    + (((upper - lower) * fraction
                        + INT64_C(128)) >> 8);

                output_row[j] = (int32_t)interpolated;
            }

            for (size_t j = t + 1; j < token_count; ++j) {
                output_row[j] = 0;
            }
        }
    }
}


void attention_normalize_exp_i32(
    const int32_t *exponentials,
    int32_t *probabilities,
    size_t head_count,
    size_t token_count,
    unsigned int probability_frac_bits
)
{
    const int64_t probability_one =
        INT64_C(1) << probability_frac_bits;

    for (size_t h = 0; h < head_count; ++h) {
        for (size_t t = 0; t < token_count; ++t) {
            const size_t row_offset =
                (h * token_count + t) * token_count;

            const int32_t *exp_row = exponentials + row_offset;
            int32_t *prob_row = probabilities + row_offset;

            int64_t denominator = 0;

            /* Finish reading the sum before overwriting any values. */
            for (size_t j = 0; j <= t; ++j) {
                denominator += (int64_t)exp_row[j];
            }

            const int64_t rounding = denominator / 2;

            for (size_t j = 0; j <= t; ++j) {
                const int64_t numerator =
                    (int64_t)exp_row[j] * probability_one;

                /* Positive division: nearest, ties toward +infinity. */
                prob_row[j] = (int32_t)(
                    (numerator + rounding) / denominator
                );
            }

            for (size_t j = t + 1; j < token_count; ++j) {
                prob_row[j] = 0;
            }
        }
    }
}

void attention_context_i8(
    const int32_t *probabilities,
    const int8_t *value,
    int32_t *context,
    size_t token_count,
    size_t head_count,
    size_t head_dim,
    int32_t value_zero_point
)
{
    const size_t hidden_size = head_count * head_dim;

    for (size_t h = 0; h < head_count; ++h) {
        const size_t head_offset = h * head_dim;

        for (size_t t = 0; t < token_count; ++t) {
            const int32_t *probability_row =
                probabilities
                + (h * token_count + t) * token_count;

            int32_t *context_head =
                context + t * hidden_size + head_offset;

            for (size_t d = 0; d < head_dim; ++d) {
                int32_t accumulator = 0;

                for (size_t j = 0; j < token_count; ++j) {
                    const int32_t centered_value =
                        (int32_t)value[
                            j * hidden_size + head_offset + d
                        ] - value_zero_point;

                    accumulator +=
                        probability_row[j] * centered_value;
                }

                context_head[d] = accumulator;
            }
        }
    }
}