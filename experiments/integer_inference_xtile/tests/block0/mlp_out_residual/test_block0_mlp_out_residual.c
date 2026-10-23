#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../../../src/blocks/block0.h"
#include "../../../src/projection.h"
#include "../../../src/requantization.h"
#include "../../../src/residual.h"
#include "../../../params/block0_params.h"
#include "../../../fixtures/block0/mlp_out_residual_golden.h"

#define INPUT_ELEMENTS BLOCK0_MLP_OUT_RESIDUAL_INPUT_ELEMENTS
#define ELEMENTS BLOCK0_MLP_OUT_RESIDUAL_ELEMENTS

_Static_assert(
    INPUT_ELEMENTS
        == BLOCK0_TOKEN_COUNT * BLOCK0_MLP_OUT_INPUT_FEATURES,
    "Input shape mismatch"
);

_Static_assert(
    ELEMENTS == BLOCK0_TOKEN_COUNT * BLOCK0_HIDDEN_SIZE,
    "Output shape mismatch"
);

_Static_assert(
    BLOCK0_MLP_OUT_OUTPUT_FEATURES == BLOCK0_HIDDEN_SIZE,
    "Projection output feature mismatch"
);

_Static_assert(
    BLOCK0_MLP_OUT_INPUT_FEATURES
        == BLOCK0_MLP_EXPAND_OUTPUT_FEATURES,
    "MLP feature mismatch"
);

_Static_assert(
    BLOCK0_RESIDUAL2_FRAC_BITS == 16u,
    "Golden T106/T107 use scale 2^-16"
);

#define DEFINE_CHECK(name, type)                                        \
static int name(                                                       \
    unsigned int c, const char *path, const char *stage,                 \
    const type *actual, const type *expected, size_t count               \
)                                                                      \
{                                                                      \
    for (size_t i = 0; i < count; ++i) {                                 \
        if (actual[i] != expected[i]) {                                 \
            printf(                                                    \
                "FAIL case %u %s %s index %u: "                         \
                "got %" PRId64 ", expected %" PRId64 "\n",              \
                c, path, stage, (unsigned int)i,                        \
                (int64_t)actual[i], (int64_t)expected[i]                 \
            );                                                         \
            return 1;                                                  \
        }                                                              \
    }                                                                  \
    return 0;                                                          \
}

DEFINE_CHECK(check_i8, int8_t)
DEFINE_CHECK(check_i32, int32_t)

#undef DEFINE_CHECK

static int32_t mean_sum[BLOCK0_TOKEN_COUNT];
static int32_t variance[BLOCK0_TOKEN_COUNT];
static int32_t inv_std[BLOCK0_TOKEN_COUNT];

static int32_t product_workspace[INPUT_ELEMENTS];
static int64_t fine_workspace[BLOCK0_GELU_ADD_CHUNK_ELEMENTS];

_Static_assert(
    BLOCK0_GELU_ADD_CHUNK_ELEMENTS >= ELEMENTS,
    "Shared fine workspace must also hold LN intermediates"
);

/* Attention scores, then the retained T73 residual. */
static int32_t score_workspace[BLOCK0_SCORE_ELEMENTS];

static int8_t ln_output[ELEMENTS];
static int8_t query[ELEMENTS];
static int8_t key[ELEMENTS];
static int8_t value[ELEMENTS];
static int8_t attention_output[ELEMENTS];
static int8_t projection_output[ELEMENTS];

static int8_t t89[INPUT_ELEMENTS];
static int8_t half[INPUT_ELEMENTS];
static int8_t branch[INPUT_ELEMENTS];
static int8_t gelu_output[INPUT_ELEMENTS];

static int32_t final_output[ELEMENTS];

static LayerNormI8Workspace ln_workspace = {
    .mean_sum = mean_sum,
    .variance = variance,
    .inv_std = inv_std,
    .feature_i32 = product_workspace,
    .fine_i64 = fine_workspace
};

static Block0MlpGeluWorkspace mlp_workspace = {
    .t89 = t89,
    .half = half,
    .branch = branch,
    .product_i32 = product_workspace,
    .wide_i64 = fine_workspace
};

#define CHECK(kind, stage, actual, expected, count)                     \
    do {                                                               \
        if (check_##kind(                                              \
            c, path, stage, actual, expected, count                     \
        )) {                                                           \
            return 1;                                                  \
        }                                                              \
    } while (0)

static int test_part7(
    unsigned int c,
    const char *path,
    const int8_t *input,
    const int32_t *residual,
    const Block0MlpOutResidualGolden *golden
)
{
    CHECK(i8, "T104 input", input, golden->t104, INPUT_ELEMENTS);
    CHECK(i32, "T73 residual", residual, golden->t73, ELEMENTS);

    projection_accumulate_i8(
        input,
        block0_mlp_out_weights,
        block0_mlp_out_bias,
        product_workspace,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_MLP_OUT_INPUT_FEATURES,
        BLOCK0_MLP_OUT_OUTPUT_FEATURES,
        BLOCK0_MLP_OUT_INPUT_ZERO_POINT
    );

    CHECK(
        i32, "Op71 accumulator",
        product_workspace, golden->accumulator, ELEMENTS
    );

    requantize_i32_to_i8(
        product_workspace,
        projection_output,
        ELEMENTS,
        BLOCK0_MLP_OUT_MULTIPLIER,
        BLOCK0_MLP_OUT_SHIFT,
        BLOCK0_MLP_OUT_OUTPUT_ZERO_POINT
    );

    CHECK(
        i8, "T105", projection_output, golden->t105, ELEMENTS
    );

    requantize_i8_to_i32(
        projection_output,
        product_workspace,
        ELEMENTS,
        BLOCK0_RESIDUAL2_BRANCH_MULTIPLIER,
        BLOCK0_RESIDUAL2_BRANCH_SHIFT,
        BLOCK0_MLP_OUT_OUTPUT_ZERO_POINT
    );

    CHECK(
        i32, "T106", product_workspace, golden->t106, ELEMENTS
    );

    /* Check addition while overwriting the T106 buffer. */
    residual_add_i32(
        product_workspace,
        residual,
        product_workspace,
        ELEMENTS
    );

    CHECK(
        i32, "T107 checkpoints",
        product_workspace, golden->t107, ELEMENTS
    );

    /*
     * Check the full wrapper with output overwriting a copy of T73.
     * The original residual remains available for comparison.
     */
    for (size_t i = 0; i < ELEMENTS; ++i) {
        final_output[i] = residual[i];
    }

    block0_mlp_out_residual_forward(
        input,
        final_output,
        final_output,
        projection_output,
        product_workspace
    );

    CHECK(
        i32, "T107 forward",
        final_output, golden->t107, ELEMENTS
    );

    CHECK(
        i32, "original T73 preserved",
        residual, golden->t73, ELEMENTS
    );

    return 0;
}

#undef CHECK

int main(void)
{
    for (unsigned int c = 0;
         c < BLOCK0_MLP_OUT_RESIDUAL_CASE_COUNT;
         ++c) {
        const Block0MlpOutResidualGolden *local =
            &block0_mlp_out_residual_local[c];

        if (test_part7(
            c, "local", local->t104, local->t73, local
        )) {
            return 1;
        }

        /* Complete C chain from quantized T0 through Part 6. */
        const int8_t *t0 = block0_mlp_out_residual_t0[c];

        block0_ln1_forward(t0, ln_output, &ln_workspace);

        block0_qkv_forward(
            ln_output, query, key, value, product_workspace
        );

        block0_attention_forward(
            query,
            key,
            value,
            attention_output,
            score_workspace,
            product_workspace
        );

        block0_out_residual_forward(
            attention_output,
            t0,
            score_workspace,
            projection_output,
            product_workspace
        );

        block0_ln2_forward(
            score_workspace, ln_output, &ln_workspace
        );

        block0_mlp_gelu_forward(
            ln_output, gelu_output, &mlp_workspace
        );

        /*
         * score_workspace must still contain T73 after Part 6.
         * test_part7 checks it against the chained golden.
         */
        if (test_part7(
            c,
            "chained",
            gelu_output,
            score_workspace,
            &block0_mlp_out_residual_chained[c]
        )) {
            return 1;
        }

        printf(
            "Case %u: local and LN1-chained "
            "Part 7 checkpoints and forward PASS\n",
            c
        );
    }

    printf(
        "PASS: %u cases, %u T107 outputs per path, 0 mismatches\n",
        (unsigned int)BLOCK0_MLP_OUT_RESIDUAL_CASE_COUNT,
        (unsigned int)(
            BLOCK0_MLP_OUT_RESIDUAL_CASE_COUNT * ELEMENTS
        )
    );

    return 0;
}