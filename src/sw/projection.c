#include "projection.h"
#include "xtile.h"

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
    /* No output elements to compute. */
    if (token_count == 0 || output_features == 0) {
        return;
    }

    /*
     * Validate before packing the direct 10-bit dimensions.
     * This implementation requires K >= 1.
     */
    if (token_count > 1023 ||
        output_features > 1023 ||
        input_features == 0 ||
        input_features > 1023) {
        __builtin_trap();
    }

    /*
     * INT8 × INT8 -> INT32.
     * Active tile: 8 × 8.
     *
     * A:  [token_count, input_features]
     * BT: [output_features, input_features]
     * C:  [token_count, output_features]
     */
    const uint64_t cfg =
        (UINT64_C(7) << 2) |
        (UINT64_C(7) << 5) |
        ((uint64_t)token_count << 8) |
        ((uint64_t)output_features << 18) |
        ((uint64_t)input_features << 28);

    xtile_cfg(cfg);
    xtile_set_ab(input, weights);
    xtile_set_c(output);
    xtile_set_zp(input_zero_point, 0);
    xtile_run();

    /*
     * RUN excludes bias.
     * Bias addition must fit INT32, as required by projection.h.
     */
    if (bias != NULL) {
        for (size_t t = 0; t < token_count; ++t) {
            int32_t *output_row =
                output + t * output_features;

            for (size_t o = 0; o < output_features; ++o) {
                output_row[o] += bias[o];
            }
        }
    }
}