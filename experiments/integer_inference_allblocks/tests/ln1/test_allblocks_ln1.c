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
#include "ln1_golden.h"

_Static_assert(LN1_TOKENS == 32u && LN1_FEATURES == 128u, "Wrong LN1 shape");

typedef void (*Ln1Forward)(const int8_t *, int8_t *, LayerNormI8Workspace *);
typedef struct {
    Ln1Forward forward;
    const LayerNormI8Params *params;
    const Ln1Golden *cases;
} BlockEntry;

static const BlockEntry blocks[] = {
    {block0_ln1_forward, &block0_ln1_params, block0_ln1_cases},
    {block1_ln1_forward, &block1_ln1_params, block1_ln1_cases},
    {block2_ln1_forward, &block2_ln1_params, block2_ln1_cases},
    {block3_ln1_forward, &block3_ln1_params, block3_ln1_cases},
    {block4_ln1_forward, &block4_ln1_params, block4_ln1_cases},
    {block5_ln1_forward, &block5_ln1_params, block5_ln1_cases},
    {block6_ln1_forward, &block6_ln1_params, block6_ln1_cases},
    {block7_ln1_forward, &block7_ln1_params, block7_ln1_cases},
};

/* Aligned guards surround every writable input, output and workspace array. */
#define GUARDED_TYPE(name, type, count) \
    typedef struct { uint64_t before; type data[count]; uint64_t after; } name
GUARDED_TYPE(GuardI8, int8_t, LN1_ELEMENTS);
GUARDED_TYPE(GuardRow, int32_t, LN1_TOKENS);
GUARDED_TYPE(GuardI32, int32_t, LN1_ELEMENTS);
GUARDED_TYPE(GuardI64, int64_t, LN1_ELEMENTS);

static GuardI8 input, output;
static GuardRow mean_sum, variance, inv_std;
static GuardI32 feature;
static GuardI64 fine;
static LayerNormI8Workspace workspace = {
    mean_sum.data, variance.data, inv_std.data, feature.data, fine.data
};
#define CANARY UINT64_C(0x7ac29e13b6054df8)
#define SET_GUARD(buffer) do { (buffer).before = CANARY; (buffer).after = ~CANARY; } while (0)

#define CHECK(label, actual, expected, count) \
    do { \
        for (size_t check_i = 0; check_i < (size_t)(count); ++check_i) { \
            const int64_t got = (int64_t)(actual)[check_i]; \
            const int64_t want = (int64_t)(expected)[check_i]; \
            if (got != want) { \
                printf("FAIL block %u case %u (%s), %s[%u]: got %" PRId64 \
                       ", expected %" PRId64 "\n", block_index, case_index, \
                       ln1_case_names[case_index], label, (unsigned int)check_i, got, want); \
                return 1; \
            } \
        } \
    } while (0)

#define CHECK_GUARD(buffer) \
    do { \
        if ((buffer).before != CANARY || (buffer).after != ~CANARY) { \
            printf("FAIL block %u case %u: %s guard changed\n", \
                   block_index, case_index, #buffer); \
            return 1; \
        } \
    } while (0)

static int check_guards(unsigned int block_index, unsigned int case_index)
{
    CHECK_GUARD(input);
    CHECK_GUARD(output);
    CHECK_GUARD(mean_sum);
    CHECK_GUARD(variance);
    CHECK_GUARD(inv_std);
    CHECK_GUARD(feature);
    CHECK_GUARD(fine);
    return 0;
}

static int test_case(unsigned int block_index, unsigned int case_index)
{
    const BlockEntry *block = &blocks[block_index];
    const LayerNormI8Params *p = block->params;
    const Ln1Golden *g = &block->cases[case_index];
    SET_GUARD(input); SET_GUARD(output); SET_GUARD(mean_sum);
    SET_GUARD(variance); SET_GUARD(inv_std); SET_GUARD(feature); SET_GUARD(fine);
    for (size_t i = 0; i < LN1_ELEMENTS; ++i) {
        input.data[i] = g->input[i];
        output.data[i] = 93;
        feature.data[i] = INT32_C(1234567);
        fine.data[i] = INT64_C(1234567890123);
    }
    for (size_t i = 0; i < LN1_TOKENS; ++i) {
        mean_sum.data[i] = -12345;
        variance.data[i] = -12345;
        inv_std.data[i] = -12345;
    }

    /* Actual block-specific orchestration, including its three XREQUANT calls. */
    block->forward(input.data, output.data, &workspace);
    CHECK("forward output", output.data, g->output, LN1_ELEMENTS);
    CHECK("forward mean", mean_sum.data, g->mean, LN1_TOKENS);
    CHECK("forward variance+epsilon", variance.data, g->variance_eps, LN1_TOKENS);
    CHECK("forward inv_std", inv_std.data, g->inv_std, LN1_TOKENS);
    CHECK("forward normalized", feature.data, g->normalized, LN1_ELEMENTS);
    CHECK("forward affine", fine.data, g->affine, LN1_ELEMENTS);
    if (check_guards(block_index, case_index)) return 1;

    /* Individual checkpoints isolate arithmetic, conversion or wiring failures. */
    layernorm_sum_i8(input.data, mean_sum.data, LN1_TOKENS, LN1_FEATURES);
    CHECK("mean", mean_sum.data, g->mean, LN1_TOKENS);
    layernorm_squared_difference_i8(input.data, mean_sum.data, fine.data, LN1_TOKENS, LN1_FEATURES);
    CHECK("square fine", fine.data, g->square_fine, LN1_ELEMENTS);
    requantize_shift_i64_to_i32(fine.data, feature.data, LN1_ELEMENTS, p->square_shift);
    CHECK("square stored", feature.data, g->square, LN1_ELEMENTS);
    layernorm_variance_i32(feature.data, variance.data, LN1_TOKENS, LN1_FEATURES);
    CHECK("variance", variance.data, g->variance, LN1_TOKENS);
    layernorm_add_epsilon_i32(variance.data, variance.data, LN1_TOKENS, p->epsilon_q);
    CHECK("variance+epsilon", variance.data, g->variance_eps, LN1_TOKENS);
    layernorm_rsqrt_i32(variance.data, inv_std.data, LN1_TOKENS, p->rsqrt_numerator);
    CHECK("inv_std", inv_std.data, g->inv_std, LN1_TOKENS);
    layernorm_normalize_i8(input.data, mean_sum.data, inv_std.data, fine.data, LN1_TOKENS, LN1_FEATURES);
    CHECK("normalized fine", fine.data, g->normalized_fine, LN1_ELEMENTS);
    requantize_shift_i64_to_i32(fine.data, feature.data, LN1_ELEMENTS, p->normalize_shift);
    CHECK("normalized stored", feature.data, g->normalized, LN1_ELEMENTS);
    layernorm_affine_i32(feature.data, p->gamma, p->beta, fine.data, LN1_TOKENS, LN1_FEATURES, p->beta_alignment);
    CHECK("affine", fine.data, g->affine, LN1_ELEMENTS);
    requantize_i64_to_i8(fine.data, output.data, LN1_ELEMENTS, p->output_multiplier, p->output_shift, p->output_zero_point);
    CHECK("staged output", output.data, g->output, LN1_ELEMENTS);
    CHECK("input unchanged", input.data, g->input, LN1_ELEMENTS);
    return check_guards(block_index, case_index);
}

int main(void)
{
    for (unsigned int b = 0; b < 8; ++b) {
        for (unsigned int c = 0; c < LN1_CASE_COUNT; ++c) {
            if (test_case(b, c)) return 1;
        }
        printf("Block %u: %u LN1 cases, all checkpoints and forward PASS\n", b, LN1_CASE_COUNT);
    }
    printf("PASS: 8 blocks, %u LN1 cases, %u outputs per path, 0 mismatches; guards unchanged\n",
           8u * LN1_CASE_COUNT, 8u * LN1_CASE_COUNT * LN1_ELEMENTS);
    return 0;
}
