#include <inttypes.h>
#include <stdio.h>

#include "../../../src/blocks/block0.h"
#include "../../../fixtures/block0/ln1_sum_data.h"
#include "../../../fixtures/block0/qkv_golden.h"
#include "../../../fixtures/block0/attention_scores_golden.h"

#define TOKENS BLOCK0_TOKEN_COUNT
#define FEATURES BLOCK0_HIDDEN_SIZE
#define ELEMENTS (TOKENS * FEATURES)
#define SCORE_ELEMENTS BLOCK0_SCORE_ELEMENTS

_Static_assert(FEATURES == BLOCK0_HEAD_COUNT * BLOCK0_HEAD_DIM,
               "Head dimensions do not match hidden size");
_Static_assert(SCORE_ELEMENTS == BLOCK0_ATTN_SCORE_ELEMENTS,
               "Score shape mismatch");
_Static_assert(BLOCK0_LN1_CASE_COUNT == BLOCK0_ATTN_SCORE_CASE_COUNT,
               "LN1 case count mismatch");
_Static_assert(BLOCK0_QKV_CASE_COUNT == BLOCK0_ATTN_SCORE_CASE_COUNT,
               "QKV case count mismatch");

static int32_t mean_sum[TOKENS];
static int32_t variance[TOKENS];
static int32_t inv_std[TOKENS];
static int32_t feature_i32[ELEMENTS];
static int64_t fine_i64[ELEMENTS];

static int8_t t45[ELEMENTS];
static int8_t q_output[ELEMENTS];
static int8_t k_output[ELEMENTS];
static int8_t v_output[ELEMENTS];
static int32_t scores[SCORE_ELEMENTS];

static LayerNormI8Workspace ln1_workspace = {
    .mean_sum = mean_sum,
    .variance = variance,
    .inv_std = inv_std,
    .feature_i32 = feature_i32,
    .fine_i64 = fine_i64
};

#define DEFINE_COMPARE(function_name, type)                          \
    static int function_name(                                       \
        size_t case_index, const char *label,                       \
        const type *actual, const type *expected, size_t count      \
    )                                                               \
    {                                                               \
        for (size_t i = 0; i < count; ++i) {                         \
            const int64_t got = (int64_t)actual[i];                  \
            const int64_t want = (int64_t)expected[i];               \
            if (got != want) {                                     \
                printf("FAIL case %u, %s[%u]: "                     \
                       "got %" PRId64 ", expected %" PRId64 "\n",   \
                       (unsigned int)case_index, label,             \
                       (unsigned int)i, got, want);                 \
                return 1;                                          \
            }                                                      \
        }                                                          \
        return 0;                                                  \
    }

DEFINE_COMPARE(compare_i8, int8_t)
DEFINE_COMPARE(compare_i32, int32_t)

int main(void)
{
    for (size_t c = 0; c < BLOCK0_ATTN_SCORE_CASE_COUNT; ++c) {
        /* Local: saved Q/K from reference T45. */
        block0_attention_scores_forward(
            block0_q_output_local_golden[c],
            block0_k_output_local_golden[c],
            scores
        );

        if (compare_i32(
                c, "local scores", scores,
                block0_attention_scores_local_golden[c],
                SCORE_ELEMENTS)) {
            return 1;
        }

        /* Chained: T0 -> C LN1 -> C Q/K/V -> C scores. */
        block0_ln1_forward(
            block0_ln1_t0[c], t45, &ln1_workspace
        );

        if (compare_i8(
                c, "chained T45", t45,
                block0_qkv_input_chained[c], ELEMENTS)) {
            return 1;
        }

        /* Reuse LN1's INT32 feature buffer for accumulators. */
        block0_qkv_forward(
            t45, q_output, k_output, v_output, feature_i32
        );

        if (compare_i8(
                c, "chained Q", q_output,
                block0_q_output_chained_golden[c], ELEMENTS)) {
            return 1;
        }

        if (compare_i8(
                c, "chained K", k_output,
                block0_k_output_chained_golden[c], ELEMENTS)) {
            return 1;
        }

        block0_attention_scores_forward(
            q_output, k_output, scores
        );

        if (compare_i32(
                c, "chained scores", scores,
                block0_attention_scores_chained_golden[c],
                SCORE_ELEMENTS)) {
            return 1;
        }

        printf("Case %u: local and LN1-chained scores PASS\n",
               (unsigned int)c);
    }

    printf("PASS: %u cases, %u scores per path, 0 mismatches\n",
           (unsigned int)BLOCK0_ATTN_SCORE_CASE_COUNT,
           (unsigned int)(
               BLOCK0_ATTN_SCORE_CASE_COUNT * SCORE_ELEMENTS
           ));

    return 0;
}