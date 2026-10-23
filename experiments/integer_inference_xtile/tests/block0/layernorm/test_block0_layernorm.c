#include <inttypes.h>
#include <stdio.h>

#include "../../../src/blocks/block0.h"
#include "../../../src/requantization.h"
#include "../../../params/block0_params.h"
#include "../../../fixtures/block0/ln1_sum_data.h"
#include "../../../fixtures/block0/ln1_golden.h"

#define TOKENS BLOCK0_TOKEN_COUNT
#define FEATURES BLOCK0_HIDDEN_SIZE
#define ELEMENTS (TOKENS * FEATURES)

_Static_assert(TOKENS == BLOCK0_LN1_TOKEN_COUNT, "Token mismatch");
_Static_assert(FEATURES == BLOCK0_LN1_FEATURE_COUNT, "Feature mismatch");

static int32_t mean_sum[TOKENS];
static int32_t variance[TOKENS];
static int32_t inv_std[TOKENS];
static int32_t feature_i32[ELEMENTS];
static int64_t fine_i64[ELEMENTS];
static int8_t output[ELEMENTS];

static LayerNormI8Workspace workspace = {
    .mean_sum = mean_sum,
    .variance = variance,
    .inv_std = inv_std,
    .feature_i32 = feature_i32,
    .fine_i64 = fine_i64
};

/* Compare integer values without narrowing either operand. */
#define CHECK(label, actual, expected, count)                     \
    do {                                                         \
        for (size_t i = 0; i < (size_t)(count); ++i) {             \
            const int64_t got = (int64_t)(actual)[i];             \
            const int64_t want = (int64_t)(expected)[i];           \
            if (got != want) {                                   \
                    printf("FAIL case %u, %s[%u]: "                   \
                       "got %" PRId64 ", expected %" PRId64 "\n", \
                       (unsigned int)case_index, label,           \
                       (unsigned int)i, got, want);               \
                return 1;                                        \
            }                                                    \
        }                                                        \
    } while (0)

static int test_case(size_t case_index)
{
    const int8_t *input = block0_ln1_t0[case_index];
    const LayerNormI8Params *p = &block0_ln1_params;

    /* First check the actual block-0 orchestration function. */
    block0_ln1_forward(input, output, &workspace);

    CHECK("forward T45", output,
          block0_ln1_t45_golden[case_index], ELEMENTS);

    /*
     * Then run the individual stages, checking each result
     * before its workspace buffer is reused.
     */
    layernorm_sum_i8(input, mean_sum, TOKENS, FEATURES);
    CHECK("T31", mean_sum,
          block0_ln1_t31_golden[case_index], TOKENS);

    layernorm_squared_difference_i8(
        input, mean_sum, fine_i64, TOKENS, FEATURES
    );
    CHECK("T33 fine", fine_i64,
          block0_ln1_t33_fine_golden[case_index], ELEMENTS);

    requantize_shift_i64_to_i32(
        fine_i64, feature_i32, ELEMENTS, p->square_shift
    );
    CHECK("T33", feature_i32,
          block0_ln1_t33_golden[case_index], ELEMENTS);

    layernorm_variance_i32(
        feature_i32, variance, TOKENS, FEATURES
    );
    CHECK("T34", variance,
          block0_ln1_t34_golden[case_index], TOKENS);

    layernorm_add_epsilon_i32(
        variance, variance, TOKENS, p->epsilon_q
    );
    CHECK("T35", variance,
          block0_ln1_t35_golden[case_index], TOKENS);

    layernorm_rsqrt_i32(
        variance, inv_std, TOKENS, p->rsqrt_numerator
    );
    CHECK("T36", inv_std,
          block0_ln1_t36_golden[case_index], TOKENS);

    layernorm_normalize_i8(
        input, mean_sum, inv_std, fine_i64, TOKENS, FEATURES
    );
    CHECK("T42 fine", fine_i64,
          block0_ln1_t42_fine_golden[case_index], ELEMENTS);

    requantize_shift_i64_to_i32(
        fine_i64, feature_i32, ELEMENTS, p->normalize_shift
    );
    CHECK("T42", feature_i32,
          block0_ln1_t42_golden[case_index], ELEMENTS);

    layernorm_affine_i32(
        feature_i32, p->gamma, p->beta, fine_i64,
        TOKENS, FEATURES, p->beta_alignment
    );
    CHECK("T44", fine_i64,
          block0_ln1_t44_golden[case_index], ELEMENTS);

    requantize_i64_to_i8(
        fine_i64, output, ELEMENTS,
        p->output_multiplier, p->output_shift,
        p->output_zero_point
    );
    CHECK("T45", output,
          block0_ln1_t45_golden[case_index], ELEMENTS);

    printf("Case %u: all checkpoints and forward output PASS\n",
       (unsigned int)case_index);
    return 0;
}

int main(void)
{
    for (size_t c = 0; c < BLOCK0_LN1_CASE_COUNT; ++c) {
        if (test_case(c) != 0) {
            return 1;
        }
    }

    printf("PASS: %u cases, %u forward INT8 outputs, %u mismatches\n",
       (unsigned int)BLOCK0_LN1_CASE_COUNT,
       (unsigned int)(BLOCK0_LN1_CASE_COUNT * ELEMENTS),
       0u);
    return 0;
}