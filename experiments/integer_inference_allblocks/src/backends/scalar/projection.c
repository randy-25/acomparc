#include "projection.h"

/* Same API, shapes, zero-point convention and bias order as the XTILE path.
 * Validated model bounds guarantee every INT32 product/partial sum/bias add.
 */
void projection_accumulate_i8(const int8_t *input, const int8_t *weights,
                              const int32_t *bias, int32_t *output,
                              size_t token_count, size_t input_features,
                              size_t output_features, int32_t input_zero_point)
{
    if (token_count == 0 || output_features == 0) return;
    if (token_count > 1023 || output_features > 1023 || input_features == 0 || input_features > 1023)
        __builtin_trap();
    for (size_t t = 0; t < token_count; ++t) {
        for (size_t o = 0; o < output_features; ++o) {
            int32_t sum = 0;
            for (size_t k = 0; k < input_features; ++k) {
                const int32_t x = (int32_t)input[t*input_features+k] - input_zero_point;
                sum += x * (int32_t)weights[o*input_features+k];
            }
            output[t*output_features+o] = sum;
        }
    }
    if (bias != NULL) {
        for (size_t t = 0; t < token_count; ++t)
            for (size_t o = 0; o < output_features; ++o)
                output[t*output_features+o] += bias[o];
    }
}
