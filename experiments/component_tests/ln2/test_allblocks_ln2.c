#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "layernorm.h"
#include "requantization.h"
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
#include "ln2_golden.h"

typedef void (*Ln1Forward)(const int8_t *, int8_t *, LayerNormI8Workspace *);
typedef void (*QkvForward)(const int8_t *, int8_t *, int8_t *, int8_t *, int32_t *);
typedef void (*AttentionForward)(const int8_t *, const int8_t *, const int8_t *,
                                 int8_t *, int32_t *, int32_t *);
typedef void (*OutResidualForward)(const int8_t *, const int8_t *, int32_t *, int8_t *, int32_t *);
typedef void (*Ln2Forward)(const int32_t *, int8_t *, LayerNormI8Workspace *);
typedef struct {
    Ln1Forward ln1;
    QkvForward qkv;
    AttentionForward attention;
    OutResidualForward out_residual;
    Ln2Forward ln2;
    const LayerNormI8Params *params;
    const Ln2Golden *cases;
} BlockEntry;
#define BLOCK_ENTRY(n) {block##n##_ln1_forward, block##n##_qkv_forward, \
    block##n##_attention_forward, block##n##_out_residual_forward, \
    block##n##_ln2_forward, &block##n##_ln2_params, block##n##_ln2_cases}
static const BlockEntry blocks[] = {
    BLOCK_ENTRY(0), BLOCK_ENTRY(1), BLOCK_ENTRY(2), BLOCK_ENTRY(3),
    BLOCK_ENTRY(4), BLOCK_ENTRY(5), BLOCK_ENTRY(6), BLOCK_ENTRY(7)
};

#define GUARDED_TYPE(name, type, count) \
    typedef struct { uint64_t before; type data[count]; uint64_t after; } name
GUARDED_TYPE(GuardI8, int8_t, LN2_ELEMENTS);
GUARDED_TYPE(GuardRow, int32_t, LN2_TOKENS);
GUARDED_TYPE(GuardI32, int32_t, LN2_ELEMENTS);
GUARDED_TYPE(GuardI64, int64_t, LN2_ELEMENTS);
GUARDED_TYPE(GuardScores, int32_t, LN2_SCORE_ELEMENTS);
static GuardI8 block_input, activation, query, key, value, attention_output, projection_output, output;
static GuardRow mean_sum, variance, inv_std;
static GuardI32 input_i32, feature;
static GuardI64 fine;
static GuardScores scores;
static LayerNormI8Workspace workspace = {
    mean_sum.data, variance.data, inv_std.data, feature.data, fine.data
};
#define CANARY UINT64_C(0x91d64e23a507bcf8)
#define SET_GUARD(buffer) do { (buffer).before = CANARY; (buffer).after = ~CANARY; } while (0)

#define CHECK(label, actual, expected, count) \
    do { \
        for (size_t check_i = 0; check_i < (size_t)(count); ++check_i) { \
            const int64_t got = (int64_t)(actual)[check_i]; \
            const int64_t want = (int64_t)(expected)[check_i]; \
            if (got != want) { \
                printf("FAIL block %u case %u (%s), %s, %s[%u]: got %" PRId64 \
                       ", expected %" PRId64 "\n", block_index, case_index, \
                       ln2_case_names[case_index], path_name, label, (unsigned int)check_i, got, want); \
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
    CHECK_GUARD(attention_output); CHECK_GUARD(projection_output); CHECK_GUARD(output);
    CHECK_GUARD(mean_sum); CHECK_GUARD(variance); CHECK_GUARD(inv_std);
    CHECK_GUARD(input_i32); CHECK_GUARD(feature); CHECK_GUARD(fine); CHECK_GUARD(scores);
    return 0;
}

static int test_case(unsigned int block_index, unsigned int case_index)
{
    const BlockEntry *block = &blocks[block_index];
    const LayerNormI8Params *p = block->params;
    const Ln2Golden *g = &block->cases[case_index];
    const char *path_name = "setup";
    SET_GUARD(block_input); SET_GUARD(activation);
    SET_GUARD(query); SET_GUARD(key); SET_GUARD(value);
    SET_GUARD(attention_output); SET_GUARD(projection_output); SET_GUARD(output);
    SET_GUARD(mean_sum); SET_GUARD(variance); SET_GUARD(inv_std);
    SET_GUARD(input_i32); SET_GUARD(feature); SET_GUARD(fine); SET_GUARD(scores);
    for (size_t i = 0; i < LN2_ELEMENTS; ++i) block_input.data[i] = g->block_input[i];

    for (unsigned int path = 0; path < 2; ++path) {
        path_name = path ? "Parts1-4-chained" : "standalone";
        const Ln2PathGolden *expected = path ? &g->chained : &g->local;
        if (path) {
            block->ln1(block_input.data, activation.data, &workspace);
            block->qkv(activation.data, query.data, key.data, value.data, feature.data);
            block->attention(query.data, key.data, value.data, attention_output.data, scores.data, feature.data);
            block->out_residual(attention_output.data, block_input.data, input_i32.data,
                                 projection_output.data, feature.data);
        } else {
            for (size_t i = 0; i < LN2_ELEMENTS; ++i) input_i32.data[i] = expected->input[i];
        }
        CHECK("LN2 input", input_i32.data, expected->input, LN2_ELEMENTS);
        for (size_t i = 0; i < LN2_ELEMENTS; ++i) {
            output.data[i] = 93; feature.data[i] = INT32_C(1234567);
            fine.data[i] = INT64_C(1234567890123);
        }
        for (size_t i = 0; i < LN2_TOKENS; ++i) {
            mean_sum.data[i] = -12345; variance.data[i] = -12345; inv_std.data[i] = -12345;
        }

        /* Actual block LN2; input is kept separate from the feature workspace. */
        block->ln2(input_i32.data, output.data, &workspace);
        CHECK("forward output", output.data, expected->output, LN2_ELEMENTS);
        CHECK("forward mean", mean_sum.data, expected->mean, LN2_TOKENS);
        CHECK("forward variance+epsilon", variance.data, expected->variance_eps, LN2_TOKENS);
        CHECK("forward inv_std", inv_std.data, expected->inv_std, LN2_TOKENS);
        CHECK("forward normalized", feature.data, expected->normalized, LN2_ELEMENTS);
        CHECK("forward affine", fine.data, expected->affine, LN2_ELEMENTS);
        if (check_guards(block_index, case_index)) return 1;

        /* Check every integer intermediate before its workspace is reused. */
        layernorm_sum_i32(input_i32.data, mean_sum.data, LN2_TOKENS, LN2_FEATURES);
        CHECK("mean", mean_sum.data, expected->mean, LN2_TOKENS);
        layernorm_squared_difference_i32(input_i32.data, mean_sum.data, fine.data, LN2_TOKENS, LN2_FEATURES);
        CHECK("square fine", fine.data, expected->square_fine, LN2_ELEMENTS);
        requantize_shift_i64_to_i32(fine.data, feature.data, LN2_ELEMENTS, p->square_shift);
        CHECK("square stored", feature.data, expected->square, LN2_ELEMENTS);
        layernorm_variance_i32(feature.data, variance.data, LN2_TOKENS, LN2_FEATURES);
        CHECK("variance", variance.data, expected->variance, LN2_TOKENS);
        layernorm_add_epsilon_i32(variance.data, variance.data, LN2_TOKENS, p->epsilon_q);
        CHECK("variance+epsilon", variance.data, expected->variance_eps, LN2_TOKENS);
        layernorm_rsqrt_i32(variance.data, inv_std.data, LN2_TOKENS, p->rsqrt_numerator);
        CHECK("inv_std", inv_std.data, expected->inv_std, LN2_TOKENS);
        layernorm_normalize_i32(input_i32.data, mean_sum.data, inv_std.data, fine.data, LN2_TOKENS, LN2_FEATURES);
        CHECK("normalized fine", fine.data, expected->normalized_fine, LN2_ELEMENTS);
        requantize_shift_i64_to_i32(fine.data, feature.data, LN2_ELEMENTS, p->normalize_shift);
        CHECK("normalized stored", feature.data, expected->normalized, LN2_ELEMENTS);
        layernorm_affine_i32(feature.data, p->gamma, p->beta, fine.data, LN2_TOKENS, LN2_FEATURES, p->beta_alignment);
        CHECK("affine", fine.data, expected->affine, LN2_ELEMENTS);
        requantize_i64_to_i8(fine.data, output.data, LN2_ELEMENTS,
                            p->output_multiplier, p->output_shift, p->output_zero_point);
        CHECK("staged output", output.data, expected->output, LN2_ELEMENTS);
        CHECK("LN2 input unchanged", input_i32.data, expected->input, LN2_ELEMENTS);
        CHECK("block input unchanged", block_input.data, g->block_input, LN2_ELEMENTS);
        if (check_guards(block_index, case_index)) return 1;
    }
    return 0;
}

int main(void)
{
    for (unsigned int b = 0; b < 8; ++b) {
        for (unsigned int c = 0; c < LN2_CASE_COUNT; ++c) {
            if (test_case(b, c)) return 1;
        }
        printf("Block %u: %u standalone and Parts1-4-chained LN2 cases PASS\n", b, LN2_CASE_COUNT);
    }
    printf("PASS: 8 blocks, %u cases per path, %u LN2 INT8 outputs per path, "
           "0 mismatches; checkpoints and guards checked\n",
           8u * LN2_CASE_COUNT, 8u * LN2_CASE_COUNT * LN2_ELEMENTS);
    return 0;
}
