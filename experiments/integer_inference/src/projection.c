#include "projection.h"

void projection_accumulate_i8(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int32_t *output,
    size_t token_count,
    size_t input_features,
    size_t output_features,
    int32_t input_zero_point
)
{
    for (size_t t = 0; t < token_count; ++t) {
        const int8_t *input_row =
            input + t * input_features;

        for (size_t o = 0; o < output_features; ++o) {
            const int8_t *weight_row =
                weights + o * input_features;

            int32_t accumulator = 0;

            for (size_t k = 0; k < input_features; ++k) {
                const int32_t centered_input =
                    (int32_t)input_row[k] - input_zero_point;

                const int32_t weight = (int32_t)weight_row[k];

                accumulator += centered_input * weight;
            }

            if (bias != NULL) {
                accumulator += bias[o];
            }

            output[t * output_features + o] = accumulator;
        }
    }
}