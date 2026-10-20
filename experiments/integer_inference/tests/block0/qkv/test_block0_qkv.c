#include <inttypes.h>
#include <stdio.h>

#include "../../../src/blocks/block0.h"
#include "../../../src/projection.h"
#include "../../../src/requantization.h"
#include "../../../params/block0_params.h"
#include "../../../fixtures/block0/ln1_sum_data.h"
#include "../../../fixtures/block0/qkv_golden.h"

#define TOKENS BLOCK0_TOKEN_COUNT
#define FEATURES BLOCK0_HIDDEN_SIZE
#define ELEMENTS (TOKENS * FEATURES)

_Static_assert(ELEMENTS == BLOCK0_QKV_ELEMENTS, "Shape mismatch");
_Static_assert(BLOCK0_LN1_CASE_COUNT == BLOCK0_QKV_CASE_COUNT,
               "Case count mismatch");

static int32_t mean_sum[TOKENS];
static int32_t variance[TOKENS];
static int32_t inv_std[TOKENS];
static int32_t feature_i32[ELEMENTS];
static int64_t fine_i64[ELEMENTS];

static int8_t t45[ELEMENTS];
static int8_t q_output[ELEMENTS];
static int8_t k_output[ELEMENTS];
static int8_t v_output[ELEMENTS];

static LayerNormI8Workspace ln1_workspace = {
    .mean_sum = mean_sum,
    .variance = variance,
    .inv_std = inv_std,
    .feature_i32 = feature_i32,
    .fine_i64 = fine_i64
};

/* Generate typed comparison helpers for integer fixtures. */
#define DEFINE_COMPARE(function_name, type)                          \
    static int function_name(                                       \
        size_t case_index, const char *path, const char *label,       \
        const type *actual, const type *expected                     \
    )                                                               \
    {                                                               \
        for (size_t i = 0; i < ELEMENTS; ++i) {                      \
            const int64_t got = (int64_t)actual[i];                   \
            const int64_t want = (int64_t)expected[i];                \
            if (got != want) {                                      \
                printf("FAIL case %u, %s, %s[%u]: "                  \
                       "got %" PRId64 ", expected %" PRId64 "\n",    \
                       (unsigned int)case_index, path, label,       \
                       (unsigned int)i, got, want);                  \
                return 1;                                           \
            }                                                       \
        }                                                           \
        return 0;                                                   \
    }

DEFINE_COMPARE(compare_i8, int8_t)
DEFINE_COMPARE(compare_i32, int32_t)

typedef struct {
    const char *name;
    const int8_t *weights;
    int64_t multiplier;
    int32_t output_zero_point;

    int8_t *output;
    const int32_t *golden_accumulator;
    const int8_t *golden_output;
} ProjectionCheck;

static int test_qkv_path(
    size_t case_index,
    const int8_t *input,
    int local
)
{
    const char *path = local ? "local" : "LN1-chained";

    const ProjectionCheck checks[3] = {
        {
            .name = "Q",
            .weights = block0_q_weights,
            .multiplier = BLOCK0_Q_MULTIPLIER,
            .output_zero_point = BLOCK0_Q_OUTPUT_ZERO_POINT,
            .output = q_output,
            .golden_accumulator = local
                ? block0_q_acc_local_golden[case_index]
                : block0_q_acc_chained_golden[case_index],
            .golden_output = local
                ? block0_q_output_local_golden[case_index]
                : block0_q_output_chained_golden[case_index]
        },
        {
            .name = "K",
            .weights = block0_k_weights,
            .multiplier = BLOCK0_K_MULTIPLIER,
            .output_zero_point = BLOCK0_K_OUTPUT_ZERO_POINT,
            .output = k_output,
            .golden_accumulator = local
                ? block0_k_acc_local_golden[case_index]
                : block0_k_acc_chained_golden[case_index],
            .golden_output = local
                ? block0_k_output_local_golden[case_index]
                : block0_k_output_chained_golden[case_index]
        },
        {
            .name = "V",
            .weights = block0_v_weights,
            .multiplier = BLOCK0_V_MULTIPLIER,
            .output_zero_point = BLOCK0_V_OUTPUT_ZERO_POINT,
            .output = v_output,
            .golden_accumulator = local
                ? block0_v_acc_local_golden[case_index]
                : block0_v_acc_chained_golden[case_index],
            .golden_output = local
                ? block0_v_output_local_golden[case_index]
                : block0_v_output_chained_golden[case_index]
        }
    };

    /*
     * LN1 has finished using feature_i32.
     * Reuse it as the Q/K/V accumulator buffer.
     */
    block0_qkv_forward(
        input, q_output, k_output, v_output, feature_i32
    );

    for (size_t p = 0; p < 3; ++p) {
        const ProjectionCheck *check = &checks[p];

        if (compare_i8(
                case_index, path, check->name,
                check->output, check->golden_output)) {
            return 1;
        }
    }

    /* Check accumulators before each is overwritten. */
    for (size_t p = 0; p < 3; ++p) {
        const ProjectionCheck *check = &checks[p];

        projection_accumulate_i8(
            input, check->weights, NULL, feature_i32,
            TOKENS, FEATURES, FEATURES,
            BLOCK0_QKV_INPUT_ZERO_POINT
        );

        if (compare_i32(
                case_index, path, check->name,
                feature_i32, check->golden_accumulator)) {
            return 1;
        }

        requantize_i32_to_i8(
            feature_i32, check->output, ELEMENTS,
            check->multiplier,
            BLOCK0_QKV_REQUANT_SHIFT,
            check->output_zero_point
        );

        if (compare_i8(
                case_index, path, check->name,
                check->output, check->golden_output)) {
            return 1;
        }
    }

    return 0;
}

int main(void)
{
    for (size_t c = 0; c < BLOCK0_QKV_CASE_COUNT; ++c) {
        /* Standalone Q/K/V from reference T45. */
        if (test_qkv_path(c, block0_qkv_input_local[c], 1)) {
            return 1;
        }

        /* Chained execution: T0 -> C LN1 -> C Q/K/V. */
        block0_ln1_forward(
            block0_ln1_t0[c], t45, &ln1_workspace
        );

        if (compare_i8(
                c, "LN1-chained", "T45",
                t45, block0_qkv_input_chained[c])) {
            return 1;
        }

        if (test_qkv_path(c, t45, 0)) {
            return 1;
        }

        printf("Case %u: local and LN1-chained Q/K/V PASS\n",
               (unsigned int)c);
    }

    printf("PASS: %u cases, %u Q/K/V outputs per path, "
           "0 mismatches\n",
           (unsigned int)BLOCK0_QKV_CASE_COUNT,
           (unsigned int)(BLOCK0_QKV_CASE_COUNT * 3u * ELEMENTS));

    return 0;
}