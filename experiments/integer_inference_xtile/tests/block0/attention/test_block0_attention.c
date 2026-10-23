#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "../../../src/attention.h"
#include "../../../src/requantization.h"
#include "../../../src/blocks/block0.h"
#include "../../../params/block0_params.h"
#include "../../../fixtures/block0/qkv_golden.h"
#include "../../../fixtures/block0/attention_scores_golden.h"
#include "../../../fixtures/block0/attention_golden.h"

_Static_assert(
    BLOCK0_ATTENTION_CASE_COUNT == BLOCK0_QKV_CASE_COUNT,
    "Attention and QKV case counts differ"
);

_Static_assert(
    BLOCK0_ATTENTION_CASE_COUNT == BLOCK0_ATTN_SCORE_CASE_COUNT,
    "Attention and score case counts differ"
);

_Static_assert(
    BLOCK0_ATTENTION_SCORE_ELEMENTS == BLOCK0_SCORE_ELEMENTS,
    "Score sizes differ"
);

_Static_assert(
    BLOCK0_ATTENTION_FEATURE_ELEMENTS
        == BLOCK0_TOKEN_COUNT * BLOCK0_HIDDEN_SIZE,
    "Feature sizes differ"
);

_Static_assert(
    BLOCK0_SOFTMAX_PROB_FRAC_BITS == 16u,
    "Golden probabilities use scale 2^-16"
);

#define DEFINE_CHECK(function_name, element_type)                       \
static int function_name(                                              \
    unsigned int case_index,                                           \
    const char *path,                                                  \
    const char *stage,                                                 \
    const element_type *actual,                                        \
    const element_type *expected,                                      \
    size_t count                                                       \
)                                                                      \
{                                                                      \
    for (size_t i = 0; i < count; ++i) {                                \
        if (actual[i] != expected[i]) {                                 \
            printf(                                                    \
                "FAIL case %u %s %s index %u: "                         \
                "got %" PRId64 ", expected %" PRId64 "\n",              \
                case_index, path, stage, (unsigned int)i,                \
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

static int32_t feature_workspace[BLOCK0_ATTENTION_FEATURE_ELEMENTS];
static int64_t fine_workspace[BLOCK0_ATTENTION_FEATURE_ELEMENTS];

static int8_t ln1_output[BLOCK0_ATTENTION_FEATURE_ELEMENTS];
static int8_t query[BLOCK0_ATTENTION_FEATURE_ELEMENTS];
static int8_t key[BLOCK0_ATTENTION_FEATURE_ELEMENTS];
static int8_t value[BLOCK0_ATTENTION_FEATURE_ELEMENTS];
static int8_t output[BLOCK0_ATTENTION_FEATURE_ELEMENTS];

static int32_t scores[BLOCK0_ATTENTION_SCORE_ELEMENTS];

static LayerNormI8Workspace ln1_workspace = {
    .mean_sum = mean_sum,
    .variance = variance,
    .inv_std = inv_std,
    .feature_i32 = feature_workspace,
    .fine_i64 = fine_workspace
};

typedef struct {
    const char *path;
    const int32_t *scores;
    const int32_t *shifted;
    const int32_t *exponentials;
    const int32_t *probabilities;
    const int32_t *context;
    const int8_t *output;
} AttentionGolden;

static int check_probability_contract(
    unsigned int case_index,
    const char *path,
    const int32_t *probabilities
)
{
    const int64_t one = INT64_C(65536);

    for (size_t h = 0; h < BLOCK0_HEAD_COUNT; ++h) {
        for (size_t t = 0; t < BLOCK0_TOKEN_COUNT; ++t) {
            const size_t offset =
                (h * BLOCK0_TOKEN_COUNT + t) * BLOCK0_TOKEN_COUNT;

            int64_t sum = 0;

            for (size_t j = 0; j < BLOCK0_TOKEN_COUNT; ++j) {
                const int32_t p = probabilities[offset + j];

                if (p < 0 || p > one || (j > t && p != 0)) {
                    printf(
                        "FAIL case %u %s probability contract: "
                        "h=%u t=%u j=%u\n",
                        case_index, path,
                        (unsigned int)h,
                        (unsigned int)t,
                        (unsigned int)j
                    );
                    return 1;
                }

                sum += p;
            }

            /*
             * At most 32 independently rounded probabilities:
             * total rounding error is at most 16 integer steps.
             */
            if (sum < one - 16 || sum > one + 16) {
                printf(
                    "FAIL case %u %s probability row sum: "
                    "h=%u t=%u sum=%" PRId64 "\n",
                    case_index, path,
                    (unsigned int)h,
                    (unsigned int)t,
                    sum
                );
                return 1;
            }

            if (t == 0 && probabilities[offset] != one) {
                printf(
                    "FAIL case %u %s first-query probability\n",
                    case_index, path
                );
                return 1;
            }
        }
    }

    return 0;
}

static int test_attention_path(
    unsigned int case_index,
    const int8_t *q,
    const int8_t *k,
    const int8_t *v,
    const AttentionGolden *golden
)
{
    const size_t score_count = BLOCK0_ATTENTION_SCORE_ELEMENTS;
    const size_t feature_count = BLOCK0_ATTENTION_FEATURE_ELEMENTS;

    block0_attention_scores_forward(q, k, scores);

    if (check_i32(
        case_index, golden->path, "scores",
        scores, golden->scores, score_count
    )) {
        return 1;
    }

    attention_causal_shift_i32(
        scores, scores,
        BLOCK0_HEAD_COUNT, BLOCK0_TOKEN_COUNT
    );

    if (check_i32(
        case_index, golden->path, "shifted scores",
        scores, golden->shifted, score_count
    )) {
        return 1;
    }

    attention_exp_lut_i32(
        scores, scores,
        BLOCK0_HEAD_COUNT, BLOCK0_TOKEN_COUNT,
        block0_softmax_exp_lut,
        BLOCK0_SOFTMAX_COORD_MULTIPLIER,
        BLOCK0_SOFTMAX_COORD_SHIFT
    );

    if (check_i32(
        case_index, golden->path, "exponentials",
        scores, golden->exponentials, score_count
    )) {
        return 1;
    }

    attention_normalize_exp_i32(
        scores, scores,
        BLOCK0_HEAD_COUNT, BLOCK0_TOKEN_COUNT,
        BLOCK0_SOFTMAX_PROB_FRAC_BITS
    );

    if (check_i32(
        case_index, golden->path, "probabilities",
        scores, golden->probabilities, score_count
    )) {
        return 1;
    }

    if (check_probability_contract(case_index, golden->path, scores)) {
        return 1;
    }

    attention_context_i8(
        scores, v, feature_workspace,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_HEAD_COUNT,
        BLOCK0_HEAD_DIM,
        BLOCK0_V_OUTPUT_ZERO_POINT
    );

    if (check_i32(
        case_index, golden->path, "context",
        feature_workspace, golden->context, feature_count
    )) {
        return 1;
    }

    requantize_i32_to_i8(
        feature_workspace, output, feature_count,
        BLOCK0_ATTN_CONTEXT_MULTIPLIER,
        BLOCK0_ATTN_CONTEXT_SHIFT,
        BLOCK0_ATTN_CONTEXT_OUTPUT_ZERO_POINT
    );

    if (check_i8(
        case_index, golden->path, "T70 checkpoints",
        output, golden->output, feature_count
    )) {
        return 1;
    }

    /* Also verify the complete block-specific orchestration. */
    block0_attention_forward(
        q, k, v, output, scores, feature_workspace
    );

    return check_i8(
        case_index, golden->path, "T70 forward",
        output, golden->output, feature_count
    );
}

int main(void)
{
    for (unsigned int c = 0; c < BLOCK0_ATTENTION_CASE_COUNT; ++c) {
        const AttentionGolden local = {
            .path = "local",
            .scores = block0_attention_scores_local_golden[c],
            .shifted = block0_attention_shifted_local_golden[c],
            .exponentials = block0_attention_exp_local_golden[c],
            .probabilities =
                block0_attention_probabilities_local_golden[c],
            .context = block0_attention_context_local_golden[c],
            .output = block0_attention_output_local_golden[c]
        };

        const AttentionGolden chained = {
            .path = "chained",
            .scores = block0_attention_scores_chained_golden[c],
            .shifted = block0_attention_shifted_chained_golden[c],
            .exponentials = block0_attention_exp_chained_golden[c],
            .probabilities =
                block0_attention_probabilities_chained_golden[c],
            .context = block0_attention_context_chained_golden[c],
            .output = block0_attention_output_chained_golden[c]
        };

        /* Local path: use the previously exported Q/K/V values. */
        if (test_attention_path(
            c,
            block0_q_output_local_golden[c],
            block0_k_output_local_golden[c],
            block0_v_output_local_golden[c],
            &local
        )) {
            return 1;
        }

        /* Chained path: calculate LN1 and QKV in C first. */
        block0_ln1_forward(
            block0_attention_t0[c],
            ln1_output,
            &ln1_workspace
        );

        if (check_i8(
            c, "chained", "LN1",
            ln1_output,
            block0_qkv_input_chained[c],
            BLOCK0_ATTENTION_FEATURE_ELEMENTS
        )) {
            return 1;
        }

        block0_qkv_forward(
            ln1_output, query, key, value, feature_workspace
        );

        if (check_i8(
            c, "chained", "Q", query,
            block0_q_output_chained_golden[c],
            BLOCK0_ATTENTION_FEATURE_ELEMENTS
        )) {
            return 1;
        }

        if (check_i8(
            c, "chained", "K", key,
            block0_k_output_chained_golden[c],
            BLOCK0_ATTENTION_FEATURE_ELEMENTS
        )) {
            return 1;
        }

        if (check_i8(
            c, "chained", "V", value,
            block0_v_output_chained_golden[c],
            BLOCK0_ATTENTION_FEATURE_ELEMENTS
        )) {
            return 1;
        }

        if (test_attention_path(c, query, key, value, &chained)) {
            return 1;
        }

        printf(
            "Case %u: local and LN1-chained attention "
            "checkpoints and forward PASS\n",
            c
        );
    }

    printf(
        "PASS: %u cases, %u T70 outputs per path, 0 mismatches\n",
        (unsigned int)BLOCK0_ATTENTION_CASE_COUNT,
        (unsigned int)(
            BLOCK0_ATTENTION_CASE_COUNT
            * BLOCK0_ATTENTION_FEATURE_ELEMENTS
        )
    );

    return 0;
}