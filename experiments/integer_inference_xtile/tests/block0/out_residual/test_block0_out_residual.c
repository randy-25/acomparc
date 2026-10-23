#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../../../src/blocks/block0.h"
#include "../../../src/projection.h"
#include "../../../src/requantization.h"
#include "../../../src/residual.h"
#include "../../../params/block0_params.h"
#include "../../../fixtures/block0/out_residual_golden.h"

#define ELEMENTS BLOCK0_OUT_RESIDUAL_ELEMENTS

_Static_assert(
    ELEMENTS == BLOCK0_TOKEN_COUNT * BLOCK0_HIDDEN_SIZE,
    "Fixture shape differs from block shape"
);

_Static_assert(
    BLOCK0_RESIDUAL1_FRAC_BITS == 16u,
    "Golden T72/T73 use scale 2^-16"
);

#define DEFINE_CHECK(name, type)                                        \
static int name(                                                       \
    unsigned int c, const char *path, const char *stage,                 \
    const type *actual, const type *expected                            \
)                                                                      \
{                                                                      \
    for (size_t i = 0; i < ELEMENTS; ++i) {                              \
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

static int32_t feature_workspace[ELEMENTS];
static int64_t fine_workspace[ELEMENTS];
static int32_t score_workspace[BLOCK0_SCORE_ELEMENTS];

static int8_t ln1_output[ELEMENTS];
static int8_t query[ELEMENTS];
static int8_t key[ELEMENTS];
static int8_t value[ELEMENTS];
static int8_t attention_output[ELEMENTS];
static int8_t projection_output[ELEMENTS];

static LayerNormI8Workspace ln1_workspace = {
    .mean_sum = mean_sum,
    .variance = variance,
    .inv_std = inv_std,
    .feature_i32 = feature_workspace,
    .fine_i64 = fine_workspace
};

static int test_part4(
    unsigned int c,
    const char *path,
    const int8_t *t70,
    const int8_t *t0,
    const Block0OutResidualGolden *golden
)
{
    if (check_i8(c, path, "T70 input", t70, golden->t70)) {
        return 1;
    }

    projection_accumulate_i8(
        t70,
        block0_out_weights,
        block0_out_bias,
        feature_workspace,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_HIDDEN_SIZE,
        BLOCK0_HIDDEN_SIZE,
        BLOCK0_OUT_PROJ_INPUT_ZERO_POINT
    );

    if (check_i32(
        c, path, "accumulator",
        feature_workspace, golden->accumulator
    )) {
        return 1;
    }

    requantize_i32_to_i8(
        feature_workspace,
        projection_output,
        ELEMENTS,
        BLOCK0_OUT_PROJ_MULTIPLIER,
        BLOCK0_OUT_PROJ_SHIFT,
        BLOCK0_OUT_PROJ_OUTPUT_ZERO_POINT
    );

    if (check_i8(
        c, path, "T71", projection_output, golden->t71
    )) {
        return 1;
    }

    requantize_i8_to_i32(
        projection_output,
        feature_workspace,
        ELEMENTS,
        BLOCK0_RESIDUAL1_BRANCH_MULTIPLIER,
        BLOCK0_RESIDUAL1_BRANCH_SHIFT,
        BLOCK0_OUT_PROJ_OUTPUT_ZERO_POINT
    );

    if (check_i32(
        c, path, "T72", feature_workspace, golden->t72
    )) {
        return 1;
    }

    residual_add_i8_i32(
        t0,
        feature_workspace,
        feature_workspace,
        ELEMENTS,
        BLOCK0_RESIDUAL1_INPUT_ALIGNMENT,
        BLOCK0_RESIDUAL1_INPUT_ZERO_POINT
    );

    if (check_i32(
        c, path, "T73", feature_workspace, golden->t73
    )) {
        return 1;
    }

    /* Check orchestration and accumulator/output buffer reuse. */
    block0_out_residual_forward(
        t70,
        t0,
        feature_workspace,
        projection_output,
        feature_workspace
    );

    return check_i32(
        c, path, "T73 forward", feature_workspace, golden->t73
    );
}

int main(void)
{
    for (unsigned int c = 0;
         c < BLOCK0_OUT_RESIDUAL_CASE_COUNT;
         ++c) {
        const int8_t *t0 = block0_out_residual_t0[c];

        /* Local Part 4 input is the saved graph's T70. */
        if (test_part4(
            c,
            "local",
            block0_out_residual_local[c].t70,
            t0,
            &block0_out_residual_local[c]
        )) {
            return 1;
        }

        /* Chained input is calculated through Parts 1-3 in C. */
        block0_ln1_forward(t0, ln1_output, &ln1_workspace);

        block0_qkv_forward(
            ln1_output, query, key, value, feature_workspace
        );

        block0_attention_forward(
            query,
            key,
            value,
            attention_output,
            score_workspace,
            feature_workspace
        );

        if (test_part4(
            c,
            "chained",
            attention_output,
            t0,
            &block0_out_residual_chained[c]
        )) {
            return 1;
        }

        printf(
            "Case %u: local and LN1-chained "
            "Part 4 checkpoints and forward PASS\n",
            c
        );
    }

    printf(
        "PASS: %u cases, %u T73 outputs per path, 0 mismatches\n",
        (unsigned int)BLOCK0_OUT_RESIDUAL_CASE_COUNT,
        (unsigned int)(
            BLOCK0_OUT_RESIDUAL_CASE_COUNT * ELEMENTS
        )
    );

    return 0;
}