#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "layernorm.h"
#include "projection.h"
#include "requantization.h"
#include "residual.h"
#include "block0.h"
#include "block1.h"
#include "block2.h"
#include "block3.h"
#include "block4.h"
#include "block5.h"
#include "block6.h"
#include "block7.h"
#include "block0_params.h"
#include "block1_params.h"
#include "block2_params.h"
#include "block3_params.h"
#include "block4_params.h"
#include "block5_params.h"
#include "block6_params.h"
#include "block7_params.h"
#include "out_residual_golden.h"

typedef void (*Ln1Forward)(const int8_t *, int8_t *, LayerNormI8Workspace *);
typedef void (*QkvForward)(const int8_t *, int8_t *, int8_t *, int8_t *, int32_t *);
typedef void (*AttentionForward)(const int8_t *, const int8_t *, const int8_t *,
                                 int8_t *, int32_t *, int32_t *);
typedef void (*OutResidualForward)(const int8_t *, const int8_t *, int32_t *, int8_t *, int32_t *);
typedef struct {
    Ln1Forward ln1;
    QkvForward qkv;
    AttentionForward attention;
    OutResidualForward out_residual;
    const int8_t *weights;
    const int32_t *bias;
    int32_t input_zero_point;
    int64_t projection_multiplier;
    unsigned int projection_shift;
    int32_t projection_zero_point;
    int64_t branch_multiplier;
    unsigned int branch_shift;
    int64_t input_alignment;
    int32_t residual_zero_point;
    const OutResidualGolden *cases;
} BlockEntry;

#define BLOCK_ENTRY(n) { \
    block##n##_ln1_forward, block##n##_qkv_forward, block##n##_attention_forward, \
    block##n##_out_residual_forward, block##n##_out_weights, block##n##_out_bias, \
    BLOCK##n##_OUT_PROJ_INPUT_ZERO_POINT, BLOCK##n##_OUT_PROJ_MULTIPLIER, \
    BLOCK##n##_OUT_PROJ_SHIFT, BLOCK##n##_OUT_PROJ_OUTPUT_ZERO_POINT, \
    BLOCK##n##_RESIDUAL1_BRANCH_MULTIPLIER, BLOCK##n##_RESIDUAL1_BRANCH_SHIFT, \
    BLOCK##n##_RESIDUAL1_INPUT_ALIGNMENT, BLOCK##n##_RESIDUAL1_INPUT_ZERO_POINT, \
    block##n##_out_residual_cases }
static const BlockEntry blocks[] = {
    BLOCK_ENTRY(0), BLOCK_ENTRY(1), BLOCK_ENTRY(2), BLOCK_ENTRY(3),
    BLOCK_ENTRY(4), BLOCK_ENTRY(5), BLOCK_ENTRY(6), BLOCK_ENTRY(7)
};

#define GUARDED_TYPE(name, type, count) \
    typedef struct { uint64_t before; type data[count]; uint64_t after; } name
GUARDED_TYPE(GuardI8, int8_t, OUT_RESIDUAL_ELEMENTS);
GUARDED_TYPE(GuardRow, int32_t, OUT_RESIDUAL_TOKENS);
GUARDED_TYPE(GuardI32, int32_t, OUT_RESIDUAL_ELEMENTS);
GUARDED_TYPE(GuardI64, int64_t, OUT_RESIDUAL_ELEMENTS);
GUARDED_TYPE(GuardScores, int32_t, OUT_RESIDUAL_SCORE_ELEMENTS);
static GuardI8 block_input, activation, query, key, value, attention_output, projection_output;
static GuardRow mean_sum, variance, inv_std;
static GuardI32 feature, output;
static GuardI64 fine;
static GuardScores scores;
static LayerNormI8Workspace workspace = {
    mean_sum.data, variance.data, inv_std.data, feature.data, fine.data
};
#define CANARY UINT64_C(0x6a21d38f50e974bc)
#define SET_GUARD(buffer) do { (buffer).before = CANARY; (buffer).after = ~CANARY; } while (0)

#define CHECK(label, actual, expected, count) \
    do { \
        for (size_t check_i = 0; check_i < (size_t)(count); ++check_i) { \
            const int64_t got = (int64_t)(actual)[check_i]; \
            const int64_t want = (int64_t)(expected)[check_i]; \
            if (got != want) { \
                printf("FAIL block %u case %u (%s), %s, %s[%u]: got %" PRId64 \
                       ", expected %" PRId64 "\n", block_index, case_index, \
                       out_residual_case_names[case_index], path_name, label, (unsigned int)check_i, got, want); \
                return 1; \
            } \
        } \
    } while (0)
#define CHECK_GUARD(buffer) \
    do { \
        if ((buffer).before != CANARY || (buffer).after != ~CANARY) { \
            printf("FAIL block %u case %u: %s guard changed\n", block_index, case_index, #buffer); \
            return 1; \
        } \
    } while (0)

static int check_guards(unsigned int block_index, unsigned int case_index)
{
    CHECK_GUARD(block_input); CHECK_GUARD(activation);
    CHECK_GUARD(query); CHECK_GUARD(key); CHECK_GUARD(value);
    CHECK_GUARD(attention_output); CHECK_GUARD(projection_output);
    CHECK_GUARD(mean_sum); CHECK_GUARD(variance); CHECK_GUARD(inv_std);
    CHECK_GUARD(feature); CHECK_GUARD(output); CHECK_GUARD(fine); CHECK_GUARD(scores);
    return 0;
}

static int test_case(unsigned int block_index, unsigned int case_index)
{
    const BlockEntry *block = &blocks[block_index];
    const OutResidualGolden *g = &block->cases[case_index];
    const char *path_name = "setup";
    SET_GUARD(block_input); SET_GUARD(activation);
    SET_GUARD(query); SET_GUARD(key); SET_GUARD(value);
    SET_GUARD(attention_output); SET_GUARD(projection_output);
    SET_GUARD(mean_sum); SET_GUARD(variance); SET_GUARD(inv_std);
    SET_GUARD(feature); SET_GUARD(output); SET_GUARD(fine); SET_GUARD(scores);
    for (size_t i = 0; i < OUT_RESIDUAL_ELEMENTS; ++i) block_input.data[i] = g->block_input[i];

    for (unsigned int path = 0; path < 2; ++path) {
        path_name = path ? "Parts1-3-chained" : "standalone";
        if (path) {
            block->ln1(block_input.data, activation.data, &workspace);
            CHECK("LN1", activation.data, g->ln1, OUT_RESIDUAL_ELEMENTS);
            block->qkv(activation.data, query.data, key.data, value.data, feature.data);
            CHECK("Q", query.data, g->query, OUT_RESIDUAL_ELEMENTS);
            CHECK("K", key.data, g->key, OUT_RESIDUAL_ELEMENTS);
            CHECK("V", value.data, g->value, OUT_RESIDUAL_ELEMENTS);
            block->attention(query.data, key.data, value.data, attention_output.data, scores.data, feature.data);
        } else {
            for (size_t i = 0; i < OUT_RESIDUAL_ELEMENTS; ++i) attention_output.data[i] = g->attention[i];
        }
        CHECK("attention input", attention_output.data, g->attention, OUT_RESIDUAL_ELEMENTS);
        for (size_t i = 0; i < OUT_RESIDUAL_ELEMENTS; ++i) {
            output.data[i] = -1234567; feature.data[i] = INT32_C(1234567); projection_output.data[i] = 93;
        }

        /* Actual block orchestration with distinct accumulator/output buffers. */
        block->out_residual(attention_output.data, block_input.data, output.data,
                             projection_output.data, feature.data);
        CHECK("forward biased accumulator", feature.data, g->accumulator, OUT_RESIDUAL_ELEMENTS);
        CHECK("forward INT8 projection", projection_output.data, g->projection, OUT_RESIDUAL_ELEMENTS);
        CHECK("forward residual", output.data, g->output, OUT_RESIDUAL_ELEMENTS);
        if (check_guards(block_index, case_index)) return 1;

        /* Check the bias and each saved quantization boundary separately. */
        projection_accumulate_i8(attention_output.data, block->weights, block->bias, feature.data,
                                 OUT_RESIDUAL_TOKENS, OUT_RESIDUAL_FEATURES, OUT_RESIDUAL_FEATURES,
                                 block->input_zero_point);
        CHECK("biased accumulator", feature.data, g->accumulator, OUT_RESIDUAL_ELEMENTS);
        requantize_i32_to_i8(feature.data, projection_output.data, OUT_RESIDUAL_ELEMENTS,
                             block->projection_multiplier, block->projection_shift, block->projection_zero_point);
        CHECK("INT8 projection", projection_output.data, g->projection, OUT_RESIDUAL_ELEMENTS);
        requantize_i8_to_i32(projection_output.data, feature.data, OUT_RESIDUAL_ELEMENTS,
                             block->branch_multiplier, block->branch_shift, block->projection_zero_point);
        CHECK("INT32 branch", feature.data, g->branch, OUT_RESIDUAL_ELEMENTS);
        residual_add_i8_i32(block_input.data, feature.data, feature.data, OUT_RESIDUAL_ELEMENTS,
                             block->input_alignment, block->residual_zero_point);
        CHECK("staged residual", feature.data, g->output, OUT_RESIDUAL_ELEMENTS);
        if (check_guards(block_index, case_index)) return 1;

        /* Permitted reuse: final output and INT32 accumulator are the same buffer. */
        block->out_residual(attention_output.data, block_input.data, feature.data,
                             projection_output.data, feature.data);
        CHECK("aliased-output residual", feature.data, g->output, OUT_RESIDUAL_ELEMENTS);
        CHECK("aliased-output projection", projection_output.data, g->projection, OUT_RESIDUAL_ELEMENTS);
        CHECK("attention input unchanged", attention_output.data, g->attention, OUT_RESIDUAL_ELEMENTS);
        CHECK("block input unchanged", block_input.data, g->block_input, OUT_RESIDUAL_ELEMENTS);
        if (check_guards(block_index, case_index)) return 1;
    }
    return 0;
}

int main(void)
{
    for (unsigned int b = 0; b < 8; ++b) {
        for (unsigned int c = 0; c < OUT_RESIDUAL_CASE_COUNT; ++c) {
            if (test_case(b, c)) return 1;
        }
        printf("Block %u: %u standalone and Parts1-3-chained output/residual cases PASS\n", b, OUT_RESIDUAL_CASE_COUNT);
    }
    printf("PASS: 8 blocks, %u cases per path, %u residual INT32 outputs per path, "
           "0 mismatches; checkpoints, buffer reuse and guards checked\n",
           8u * OUT_RESIDUAL_CASE_COUNT, 8u * OUT_RESIDUAL_CASE_COUNT * OUT_RESIDUAL_ELEMENTS);
    return 0;
}
