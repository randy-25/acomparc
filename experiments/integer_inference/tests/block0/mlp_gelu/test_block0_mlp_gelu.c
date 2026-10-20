#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../../../src/blocks/block0.h"
#include "../../../src/gelu.h"
#include "../../../src/projection.h"
#include "../../../src/requantization.h"
#include "../../../params/block0_params.h"
#include "../../../fixtures/block0/mlp_gelu_golden.h"

#define INPUT_ELEMENTS BLOCK0_MLP_GELU_INPUT_ELEMENTS
#define ELEMENTS BLOCK0_MLP_GELU_ELEMENTS

_Static_assert(
    INPUT_ELEMENTS == BLOCK0_TOKEN_COUNT * BLOCK0_HIDDEN_SIZE,
    "Input shape mismatch"
);

_Static_assert(
    ELEMENTS
        == BLOCK0_TOKEN_COUNT * BLOCK0_MLP_EXPAND_OUTPUT_FEATURES,
    "Output shape mismatch"
);

_Static_assert(
    BLOCK0_MLP_EXPAND_INPUT_FEATURES == BLOCK0_HIDDEN_SIZE,
    "Projection input feature mismatch"
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

/* Shared by upstream LN/projections and the larger Part 6 calculation. */
static int32_t product_workspace[ELEMENTS];
static int64_t fine_workspace[BLOCK0_GELU_ADD_CHUNK_ELEMENTS];

_Static_assert(
    BLOCK0_GELU_ADD_CHUNK_ELEMENTS >= INPUT_ELEMENTS,
    "Shared fine workspace must also hold the LN intermediates"
);

/* Holds attention scores, then retains T73. */
static int32_t score_workspace[BLOCK0_SCORE_ELEMENTS];

static int8_t ln_output[INPUT_ELEMENTS];
static int8_t query[INPUT_ELEMENTS];
static int8_t key[INPUT_ELEMENTS];
static int8_t value[INPUT_ELEMENTS];
static int8_t attention_output[INPUT_ELEMENTS];
static int8_t projection_output[INPUT_ELEMENTS];

static int8_t t89[ELEMENTS];
static int8_t half[ELEMENTS];
static int8_t branch[ELEMENTS];
static int8_t output[ELEMENTS];

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

/* All these conversions have the same INT32 -> INT8 interface. */
#define RQ32(prefix, input, destination)                                \
    requantize_i32_to_i8(                                               \
        input, destination, ELEMENTS,                                  \
        prefix##_MULTIPLIER,                                           \
        prefix##_SHIFT,                                                \
        prefix##_OUTPUT_ZERO_POINT                                     \
    )

static int test_part6(
    unsigned int c,
    const char *path,
    const int8_t *input,
    const Block0MlpGeluGolden *golden
)
{
    CHECK(i8, "T88 input", input, golden->t88, INPUT_ELEMENTS);

    projection_accumulate_i8(
        input,
        block0_mlp_expand_weights,
        block0_mlp_expand_bias,
        product_workspace,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_MLP_EXPAND_INPUT_FEATURES,
        BLOCK0_MLP_EXPAND_OUTPUT_FEATURES,
        BLOCK0_MLP_EXPAND_INPUT_ZERO_POINT
    );

    CHECK(
        i32, "Op58 accumulator",
        product_workspace, golden->accumulator, ELEMENTS
    );

    RQ32(BLOCK0_MLP_EXPAND, product_workspace, t89);
    CHECK(i8, "T89", t89, golden->t89, ELEMENTS);

    gelu_multiply_scalar_i8(
        t89, product_workspace, ELEMENTS,
        BLOCK0_GELU_HALF_INPUT_ZERO_POINT,
        BLOCK0_GELU_HALF_CENTERED_CONSTANT
    );

    RQ32(BLOCK0_GELU_HALF, product_workspace, half);
    CHECK(i8, "T92", half, golden->t92, ELEMENTS);

    gelu_cube_i8(
        t89, product_workspace, ELEMENTS,
        BLOCK0_GELU_CUBE_INPUT_ZERO_POINT
    );

    CHECK(i32, "T93", product_workspace, golden->t93, ELEMENTS);

    RQ32(BLOCK0_GELU_CUBE, product_workspace, branch);
    CHECK(i8, "T94", branch, golden->t94, ELEMENTS);

    gelu_multiply_scalar_i8(
        branch, product_workspace, ELEMENTS,
        BLOCK0_GELU_WEIGHTED_CUBE_INPUT_ZERO_POINT,
        BLOCK0_GELU_WEIGHTED_CUBE_CENTERED_CONSTANT
    );

    RQ32(BLOCK0_GELU_WEIGHTED_CUBE, product_workspace, branch);
    CHECK(i8, "T96", branch, golden->t96, ELEMENTS);

    for (size_t offset = 0; offset < ELEMENTS;
         offset += BLOCK0_GELU_ADD_CHUNK_ELEMENTS) {
        size_t chunk = ELEMENTS - offset;

        if (chunk > BLOCK0_GELU_ADD_CHUNK_ELEMENTS) {
            chunk = BLOCK0_GELU_ADD_CHUNK_ELEMENTS;
        }

        gelu_add_scaled_i8(
            t89 + offset,
            branch + offset,
            fine_workspace,
            chunk,
            BLOCK0_GELU_ADD_LINEAR_ZERO_POINT,
            BLOCK0_GELU_ADD_CUBIC_ZERO_POINT,
            BLOCK0_GELU_ADD_LINEAR_MULTIPLIER,
            BLOCK0_GELU_ADD_CUBIC_MULTIPLIER
        );

        requantize_i64_to_i8(
            fine_workspace,
            branch + offset,
            chunk,
            INT64_C(1),
            BLOCK0_GELU_ADD_SHIFT,
            BLOCK0_GELU_ADD_OUTPUT_ZERO_POINT
        );
    }

    CHECK(i8, "T97", branch, golden->t97, ELEMENTS);

    gelu_multiply_scalar_i8(
        branch, product_workspace, ELEMENTS,
        BLOCK0_GELU_ARGUMENT_INPUT_ZERO_POINT,
        BLOCK0_GELU_ARGUMENT_CENTERED_CONSTANT
    );

    RQ32(BLOCK0_GELU_ARGUMENT, product_workspace, branch);
    CHECK(i8, "T99", branch, golden->t99, ELEMENTS);

    gelu_tanh_lut_i8(
        branch, product_workspace, ELEMENTS, block0_gelu_tanh_lut
    );

    CHECK(
        i32, "T101", product_workspace, golden->t101, ELEMENTS
    );

    gelu_add_scalar_i32(
        product_workspace,
        product_workspace,
        ELEMENTS,
        BLOCK0_GELU_TANH_ONE
    );

    CHECK(
        i32, "T102", product_workspace, golden->t102, ELEMENTS
    );

    RQ32(BLOCK0_GELU_FACTOR, product_workspace, branch);
    CHECK(i8, "T103", branch, golden->t103, ELEMENTS);

    gelu_multiply_i8(
        half, branch, product_workspace, ELEMENTS,
        BLOCK0_GELU_OUTPUT_HALF_ZERO_POINT,
        BLOCK0_GELU_OUTPUT_FACTOR_ZERO_POINT
    );

    requantize_two_round_i32_to_i8(
        product_workspace,
        output,
        ELEMENTS,
        BLOCK0_GELU_OUTPUT_Q31_MULTIPLIER,
        BLOCK0_GELU_OUTPUT_SECOND_SHIFT,
        BLOCK0_GELU_OUTPUT_ZERO_POINT
    );

    CHECK(i8, "T104 checkpoints", output, golden->t104, ELEMENTS);

    /* Also test the complete Part 6 orchestration. */
    block0_mlp_gelu_forward(input, output, &mlp_workspace);

    CHECK(i8, "T104 forward", output, golden->t104, ELEMENTS);

    return 0;
}

#undef CHECK
#undef RQ32

int main(void)
{
    for (unsigned int c = 0;
         c < BLOCK0_MLP_GELU_CASE_COUNT;
         ++c) {
        if (test_part6(
            c,
            "local",
            block0_mlp_gelu_local[c].t88,
            &block0_mlp_gelu_local[c]
        )) {
            return 1;
        }

        /* Calculate the complete upstream chain in C. */
        const int8_t *t0 = block0_mlp_gelu_t0[c];

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

        /* Retain T73 in the no-longer-needed attention-score buffer. */
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

        if (test_part6(
            c,
            "chained",
            ln_output,
            &block0_mlp_gelu_chained[c]
        )) {
            return 1;
        }

        printf(
            "Case %u: local and LN1-chained "
            "Part 6 checkpoints and forward PASS\n",
            c
        );
    }

    printf(
        "PASS: %u cases, %u T104 outputs per path, 0 mismatches\n",
        (unsigned int)BLOCK0_MLP_GELU_CASE_COUNT,
        (unsigned int)(BLOCK0_MLP_GELU_CASE_COUNT * ELEMENTS)
    );

    return 0;
}