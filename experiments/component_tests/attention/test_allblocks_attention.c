#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "attention.h"
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
#include "attention_golden.h"

typedef void (*Ln1Forward)(const int8_t *, int8_t *, LayerNormI8Workspace *);
typedef void (*QkvForward)(const int8_t *, int8_t *, int8_t *, int8_t *, int32_t *);
typedef void (*ScoresForward)(const int8_t *, const int8_t *, int32_t *);
typedef void (*AttentionForward)(const int8_t *, const int8_t *, const int8_t *,
                                 int8_t *, int32_t *, int32_t *);
typedef struct {
    Ln1Forward ln1;
    QkvForward qkv;
    ScoresForward scores;
    AttentionForward attention;
    const int32_t *exp_lut;
    int64_t coordinate_multiplier;
    unsigned int coordinate_shift;
    unsigned int probability_frac_bits;
    int32_t value_zero_point;
    int64_t context_multiplier;
    unsigned int context_shift;
    int32_t context_output_zero_point;
    const AttentionGolden *cases;
} BlockEntry;

#define BLOCK_ENTRY(n) { \
    block##n##_ln1_forward, block##n##_qkv_forward, \
    block##n##_attention_scores_forward, block##n##_attention_forward, \
    block##n##_softmax_exp_lut, BLOCK##n##_SOFTMAX_COORD_MULTIPLIER, \
    BLOCK##n##_SOFTMAX_COORD_SHIFT, BLOCK##n##_SOFTMAX_PROB_FRAC_BITS, \
    BLOCK##n##_V_OUTPUT_ZERO_POINT, BLOCK##n##_ATTN_CONTEXT_MULTIPLIER, \
    BLOCK##n##_ATTN_CONTEXT_SHIFT, BLOCK##n##_ATTN_CONTEXT_OUTPUT_ZERO_POINT, \
    block##n##_attention_cases }
static const BlockEntry blocks[] = {
    BLOCK_ENTRY(0), BLOCK_ENTRY(1), BLOCK_ENTRY(2), BLOCK_ENTRY(3),
    BLOCK_ENTRY(4), BLOCK_ENTRY(5), BLOCK_ENTRY(6), BLOCK_ENTRY(7)
};

_Static_assert(ATTENTION_TOKENS == 32u && ATTENTION_HEADS == 16u && ATTENTION_HEAD_DIM == 8u,
               "Unexpected attention dimensions");

#define GUARDED_TYPE(name, type, count) \
    typedef struct { uint64_t before; type data[count]; uint64_t after; } name
GUARDED_TYPE(GuardI8, int8_t, ATTENTION_ELEMENTS);
GUARDED_TYPE(GuardRow, int32_t, ATTENTION_TOKENS);
GUARDED_TYPE(GuardI32, int32_t, ATTENTION_ELEMENTS);
GUARDED_TYPE(GuardI64, int64_t, ATTENTION_ELEMENTS);
GUARDED_TYPE(GuardScores, int32_t, ATTENTION_SCORE_ELEMENTS);
static GuardI8 block_input, activation, query, key, value, output;
static GuardRow mean_sum, variance, inv_std;
static GuardI32 feature;
static GuardI64 fine;
static GuardScores scores;
static LayerNormI8Workspace workspace = {
    mean_sum.data, variance.data, inv_std.data, feature.data, fine.data
};
#define CANARY UINT64_C(0x3fc692e7a10458bd)
#define SET_GUARD(buffer) do { (buffer).before = CANARY; (buffer).after = ~CANARY; } while (0)

#define CHECK(label, actual, expected, count) \
    do { \
        for (size_t check_i = 0; check_i < (size_t)(count); ++check_i) { \
            const int64_t got = (int64_t)(actual)[check_i]; \
            const int64_t want = (int64_t)(expected)[check_i]; \
            if (got != want) { \
                printf("FAIL block %u case %u (%s), %s, %s[%u]: got %" PRId64 \
                       ", expected %" PRId64 "\n", block_index, case_index, \
                       attention_case_names[case_index], path_name, label, (unsigned int)check_i, got, want); \
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
    CHECK_GUARD(query); CHECK_GUARD(key); CHECK_GUARD(value); CHECK_GUARD(output);
    CHECK_GUARD(mean_sum); CHECK_GUARD(variance); CHECK_GUARD(inv_std);
    CHECK_GUARD(feature); CHECK_GUARD(fine); CHECK_GUARD(scores);
    return 0;
}

static int check_probability_contract(unsigned int block_index, unsigned int case_index,
                                      const char *path_name)
{
    for (size_t h = 0; h < ATTENTION_HEADS; ++h) {
        for (size_t t = 0; t < ATTENTION_TOKENS; ++t) {
            const size_t offset = (h * ATTENTION_TOKENS + t) * ATTENTION_TOKENS;
            int64_t sum = 0;
            for (size_t j = 0; j < ATTENTION_TOKENS; ++j) {
                const int32_t p = scores.data[offset+j];
                if (p < 0 || p > 65536 || (j > t && p != 0)) {
                    printf("FAIL block %u case %u %s probability at h=%u t=%u j=%u\n",
                           block_index, case_index, path_name, (unsigned int)h, (unsigned int)t, (unsigned int)j);
                    return 1;
                }
                sum += p;
            }
            if (sum < 65536-16 || sum > 65536+16 || (t == 0 && scores.data[offset] != 65536)) {
                printf("FAIL block %u case %u %s probability row h=%u t=%u sum=%" PRId64 "\n",
                       block_index, case_index, path_name, (unsigned int)h, (unsigned int)t, sum);
                return 1;
            }
        }
    }
    return 0;
}

static int test_case(unsigned int block_index, unsigned int case_index)
{
    const BlockEntry *block = &blocks[block_index];
    const AttentionGolden *g = &block->cases[case_index];
    const char *path_name = "setup";
    SET_GUARD(block_input); SET_GUARD(activation);
    SET_GUARD(query); SET_GUARD(key); SET_GUARD(value); SET_GUARD(output);
    SET_GUARD(mean_sum); SET_GUARD(variance); SET_GUARD(inv_std);
    SET_GUARD(feature); SET_GUARD(fine); SET_GUARD(scores);
    for (size_t i = 0; i < ATTENTION_ELEMENTS; ++i) block_input.data[i] = g->block_input[i];
    if (block->probability_frac_bits != 16u) {
        printf("FAIL block %u: expected probability scale 2^-16\n", block_index);
        return 1;
    }

    for (unsigned int path = 0; path < 2; ++path) {
        path_name = path ? "LN1-QKV-chained" : "standalone";
        if (path) {
            block->ln1(block_input.data, activation.data, &workspace);
            CHECK("LN1", activation.data, g->ln1, ATTENTION_ELEMENTS);
            block->qkv(activation.data, query.data, key.data, value.data, feature.data);
        } else {
            for (size_t i = 0; i < ATTENTION_ELEMENTS; ++i) {
                query.data[i] = g->query[i]; key.data[i] = g->key[i]; value.data[i] = g->value[i];
            }
        }
        CHECK("Q", query.data, g->query, ATTENTION_ELEMENTS);
        CHECK("K", key.data, g->key, ATTENTION_ELEMENTS);
        CHECK("V", value.data, g->value, ATTENTION_ELEMENTS);
        for (size_t i = 0; i < ATTENTION_ELEMENTS; ++i) {
            output.data[i] = 93; feature.data[i] = INT32_C(1234567);
        }
        for (size_t i = 0; i < ATTENTION_SCORE_ELEMENTS; ++i) scores.data[i] = -12345;

        /* Real block attention orchestration, with in-place score workspace. */
        block->attention(query.data, key.data, value.data, output.data, scores.data, feature.data);
        CHECK("forward output", output.data, g->output, ATTENTION_ELEMENTS);
        CHECK("forward probabilities", scores.data, g->probabilities, ATTENTION_SCORE_ELEMENTS);
        CHECK("forward context", feature.data, g->context, ATTENTION_ELEMENTS);
        if (check_probability_contract(block_index, case_index, path_name)) return 1;
        if (check_guards(block_index, case_index)) return 1;

        /* Check each software stage before the score workspace is reused. */
        block->scores(query.data, key.data, scores.data);
        CHECK("scores", scores.data, g->scores, ATTENTION_SCORE_ELEMENTS);
        attention_causal_shift_i32(scores.data, scores.data, ATTENTION_HEADS, ATTENTION_TOKENS);
        CHECK("shifted scores", scores.data, g->shifted, ATTENTION_SCORE_ELEMENTS);
        attention_exp_lut_i32(scores.data, scores.data, ATTENTION_HEADS, ATTENTION_TOKENS,
                              block->exp_lut, block->coordinate_multiplier, block->coordinate_shift);
        CHECK("exponentials", scores.data, g->exponentials, ATTENTION_SCORE_ELEMENTS);
        attention_normalize_exp_i32(scores.data, scores.data, ATTENTION_HEADS, ATTENTION_TOKENS,
                                    block->probability_frac_bits);
        CHECK("probabilities", scores.data, g->probabilities, ATTENTION_SCORE_ELEMENTS);
        if (check_probability_contract(block_index, case_index, path_name)) return 1;
        attention_context_i8(scores.data, value.data, feature.data,
                              ATTENTION_TOKENS, ATTENTION_HEADS, ATTENTION_HEAD_DIM, block->value_zero_point);
        CHECK("context", feature.data, g->context, ATTENTION_ELEMENTS);
        requantize_i32_to_i8(feature.data, output.data, ATTENTION_ELEMENTS,
                             block->context_multiplier, block->context_shift, block->context_output_zero_point);
        CHECK("staged output", output.data, g->output, ATTENTION_ELEMENTS);
        CHECK("Q unchanged", query.data, g->query, ATTENTION_ELEMENTS);
        CHECK("K unchanged", key.data, g->key, ATTENTION_ELEMENTS);
        CHECK("V unchanged", value.data, g->value, ATTENTION_ELEMENTS);
        CHECK("block input unchanged", block_input.data, g->block_input, ATTENTION_ELEMENTS);
        if (check_guards(block_index, case_index)) return 1;
    }
    return 0;
}

int main(void)
{
    for (unsigned int b = 0; b < 8; ++b) {
        for (unsigned int c = 0; c < ATTENTION_CASE_COUNT; ++c) {
            if (test_case(b, c)) return 1;
        }
        printf("Block %u: %u standalone and LN1-QKV-chained attention cases PASS\n", b, ATTENTION_CASE_COUNT);
    }
    printf("PASS: 8 blocks, %u cases per path, %u attention INT8 outputs per path, "
           "0 mismatches; checkpoints, probabilities and guards checked\n",
           8u * ATTENTION_CASE_COUNT, 8u * ATTENTION_CASE_COUNT * ATTENTION_ELEMENTS);
    return 0;
}
