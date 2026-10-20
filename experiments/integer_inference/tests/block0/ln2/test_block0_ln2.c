#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../../../src/blocks/block0.h"
#include "../../../src/layernorm.h"
#include "../../../src/requantization.h"
#include "../../../params/block0_params.h"
#include "../../../fixtures/block0/ln2_golden.h"

#define ELEMENTS BLOCK0_LN2_GOLDEN_ELEMENTS

_Static_assert(
    ELEMENTS == BLOCK0_TOKEN_COUNT * BLOCK0_HIDDEN_SIZE,
    "Fixture feature count differs from block shape"
);

_Static_assert(
    BLOCK0_LN2_GOLDEN_TOKEN_COUNT == BLOCK0_TOKEN_COUNT,
    "Fixture token count differs from block shape"
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
DEFINE_CHECK(check_i64, int64_t)

#undef DEFINE_CHECK

static int32_t mean_sum[BLOCK0_TOKEN_COUNT];
static int32_t variance[BLOCK0_TOKEN_COUNT];
static int32_t inv_std[BLOCK0_TOKEN_COUNT];

static int32_t feature_workspace[ELEMENTS];
static int64_t fine_workspace[ELEMENTS];
static int32_t score_workspace[BLOCK0_SCORE_ELEMENTS];

static int8_t ln_output[ELEMENTS];
static int8_t query[ELEMENTS];
static int8_t key[ELEMENTS];
static int8_t value[ELEMENTS];
static int8_t attention_output[ELEMENTS];
static int8_t projection_output[ELEMENTS];

static LayerNormI8Workspace workspace = {
    .mean_sum = mean_sum,
    .variance = variance,
    .inv_std = inv_std,
    .feature_i32 = feature_workspace,
    .fine_i64 = fine_workspace
};

#define CHECK(kind, stage, actual, expected, count)                     \
    do {                                                               \
        if (check_##kind(                                              \
            c, path, stage, actual, expected, count                     \
        )) {                                                           \
            return 1;                                                  \
        }                                                              \
    } while (0)

static int test_ln2(
    unsigned int c,
    const char *path,
    const int32_t *input,
    const Block0Ln2Golden *golden
)
{
    const LayerNormI8Params *params = &block0_ln2_params;

    const size_t tokens = BLOCK0_TOKEN_COUNT;
    const size_t features = BLOCK0_HIDDEN_SIZE;

    CHECK(i32, "T73 input", input, golden->t73, ELEMENTS);

    layernorm_sum_i32(
        input, mean_sum, tokens, features
    );

    CHECK(i32, "T74 mean sum", mean_sum, golden->t74, tokens);

    layernorm_squared_difference_i32(
        input, mean_sum, fine_workspace, tokens, features
    );

    CHECK(
        i64, "T76 fine",
        fine_workspace, golden->t76_fine, ELEMENTS
    );

    requantize_shift_i64_to_i32(
        fine_workspace,
        feature_workspace,
        ELEMENTS,
        params->square_shift
    );

    CHECK(
        i32, "T76",
        feature_workspace, golden->t76, ELEMENTS
    );

    layernorm_variance_i32(
        feature_workspace, variance, tokens, features
    );

    CHECK(i32, "T77 variance", variance, golden->t77, tokens);

    layernorm_add_epsilon_i32(
        variance, variance, tokens, params->epsilon_q
    );

    CHECK(i32, "T78 variance + epsilon", variance, golden->t78, tokens);

    layernorm_rsqrt_i32(
        variance, inv_std, tokens, params->rsqrt_numerator
    );

    CHECK(i32, "T79 inverse std", inv_std, golden->t79, tokens);

    layernorm_normalize_i32(
        input, mean_sum, inv_std,
        fine_workspace, tokens, features
    );

    CHECK(
        i64, "T85 fine",
        fine_workspace, golden->t85_fine, ELEMENTS
    );

    requantize_shift_i64_to_i32(
        fine_workspace,
        feature_workspace,
        ELEMENTS,
        params->normalize_shift
    );

    CHECK(
        i32, "T85 normalized",
        feature_workspace, golden->t85, ELEMENTS
    );

    layernorm_affine_i32(
        feature_workspace,
        params->gamma,
        params->beta,
        fine_workspace,
        tokens,
        features,
        params->beta_alignment
    );

    CHECK(
        i64, "T87 affine",
        fine_workspace, golden->t87, ELEMENTS
    );

    requantize_i64_to_i8(
        fine_workspace,
        ln_output,
        ELEMENTS,
        params->output_multiplier,
        params->output_shift,
        params->output_zero_point
    );

    CHECK(i8, "T88 checkpoints", ln_output, golden->t88, ELEMENTS);

    /* Independently exercise the complete LN2 orchestration. */
    block0_ln2_forward(input, ln_output, &workspace);

    CHECK(i8, "T88 forward", ln_output, golden->t88, ELEMENTS);

    return 0;
}

#undef CHECK

int main(void)
{
    for (unsigned int c = 0;
         c < BLOCK0_LN2_GOLDEN_CASE_COUNT;
         ++c) {
        /* Local LN2 test using the saved INT32 input. */
        if (test_ln2(
            c,
            "local",
            block0_ln2_local[c].t73,
            &block0_ln2_local[c]
        )) {
            return 1;
        }

        /* Complete C chain from the original quantized T0. */
        const int8_t *t0 = block0_ln2_t0[c];

        block0_ln1_forward(t0, ln_output, &workspace);

        block0_qkv_forward(
            ln_output, query, key, value, feature_workspace
        );

        block0_attention_forward(
            query,
            key,
            value,
            attention_output,
            score_workspace,
            feature_workspace
        );

        /*
         * Attention no longer needs score_workspace.
         * Reuse its first 4096 INT32 elements for T73.
         *
         * Keep T73 separate from LN2's feature_workspace.
         */
        block0_out_residual_forward(
            attention_output,
            t0,
            score_workspace,
            projection_output,
            feature_workspace
        );

        if (test_ln2(
            c,
            "chained",
            score_workspace,
            &block0_ln2_chained[c]
        )) {
            return 1;
        }

        printf(
            "Case %u: local and LN1-chained "
            "LN2 checkpoints and forward PASS\n",
            c
        );
    }

    printf(
        "PASS: %u cases, %u T88 outputs per path, 0 mismatches\n",
        (unsigned int)BLOCK0_LN2_GOLDEN_CASE_COUNT,
        (unsigned int)(
            BLOCK0_LN2_GOLDEN_CASE_COUNT * ELEMENTS
        )
    );

    return 0;
}