#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "layernorm.h"
#include "projection.h"
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
#include "qkv_golden.h"

typedef void (*Ln1Forward)(const int8_t *, int8_t *, LayerNormI8Workspace *);
typedef void (*QkvForward)(const int8_t *, int8_t *, int8_t *, int8_t *, int32_t *);
typedef struct {
    Ln1Forward ln1;
    QkvForward qkv;
    const int8_t *weights[3];
    int64_t multipliers[3];
    int32_t output_zero_points[3];
    int32_t input_zero_point;
    unsigned int shift;
    const QkvGolden *cases;
} BlockEntry;

#define BLOCK_ENTRY(n) { \
    block##n##_ln1_forward, block##n##_qkv_forward, \
    {block##n##_q_weights, block##n##_k_weights, block##n##_v_weights}, \
    {BLOCK##n##_Q_MULTIPLIER, BLOCK##n##_K_MULTIPLIER, BLOCK##n##_V_MULTIPLIER}, \
    {BLOCK##n##_Q_OUTPUT_ZERO_POINT, BLOCK##n##_K_OUTPUT_ZERO_POINT, BLOCK##n##_V_OUTPUT_ZERO_POINT}, \
    BLOCK##n##_QKV_INPUT_ZERO_POINT, BLOCK##n##_QKV_REQUANT_SHIFT, block##n##_qkv_cases }

static const BlockEntry blocks[] = {
    BLOCK_ENTRY(0), BLOCK_ENTRY(1), BLOCK_ENTRY(2), BLOCK_ENTRY(3),
    BLOCK_ENTRY(4), BLOCK_ENTRY(5), BLOCK_ENTRY(6), BLOCK_ENTRY(7)
};
static const char *const projection_names[] = {"Q", "K", "V"};
static const char *const accumulator_names[] = {"Q accumulator", "K accumulator", "V accumulator"};

#define GUARDED_TYPE(name, type, count) \
    typedef struct { uint64_t before; type data[count]; uint64_t after; } name
GUARDED_TYPE(GuardI8, int8_t, QKV_ELEMENTS);
GUARDED_TYPE(GuardRow, int32_t, QKV_TOKENS);
GUARDED_TYPE(GuardI32, int32_t, QKV_ELEMENTS);
GUARDED_TYPE(GuardI64, int64_t, QKV_ELEMENTS);
static GuardI8 block_input, activation, q_output, k_output, v_output;
static GuardRow mean_sum, variance, inv_std;
static GuardI32 accumulator;
static GuardI64 fine;
static LayerNormI8Workspace workspace = {
    mean_sum.data, variance.data, inv_std.data, accumulator.data, fine.data
};
#define CANARY UINT64_C(0x5d92e6a0c7314bf8)
#define SET_GUARD(buffer) do { (buffer).before = CANARY; (buffer).after = ~CANARY; } while (0)

#define CHECK(label, actual, expected, count) \
    do { \
        for (size_t check_i = 0; check_i < (size_t)(count); ++check_i) { \
            const int64_t got = (int64_t)(actual)[check_i]; \
            const int64_t want = (int64_t)(expected)[check_i]; \
            if (got != want) { \
                printf("FAIL block %u case %u (%s), %s, %s[%u]: got %" PRId64 \
                       ", expected %" PRId64 "\n", block_index, case_index, \
                       qkv_case_names[case_index], path_name, label, (unsigned int)check_i, got, want); \
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
    CHECK_GUARD(q_output); CHECK_GUARD(k_output); CHECK_GUARD(v_output);
    CHECK_GUARD(mean_sum); CHECK_GUARD(variance); CHECK_GUARD(inv_std);
    CHECK_GUARD(accumulator); CHECK_GUARD(fine);
    return 0;
}

static int test_case(unsigned int block_index, unsigned int case_index)
{
    const BlockEntry *block = &blocks[block_index];
    const QkvGolden *g = &block->cases[case_index];
    int8_t *const outputs[] = {q_output.data, k_output.data, v_output.data};
    const char *path_name = "setup";
    SET_GUARD(block_input); SET_GUARD(activation);
    SET_GUARD(q_output); SET_GUARD(k_output); SET_GUARD(v_output);
    SET_GUARD(mean_sum); SET_GUARD(variance); SET_GUARD(inv_std);
    SET_GUARD(accumulator); SET_GUARD(fine);
    for (size_t i = 0; i < QKV_ELEMENTS; ++i) block_input.data[i] = g->block_input[i];

    for (unsigned int path = 0; path < 2; ++path) {
        path_name = path ? "LN1-chained" : "standalone";
        const QkvPathGolden *expected = path ? &g->chained : &g->local;
        if (path) {
            block->ln1(block_input.data, activation.data, &workspace);
            CHECK("LN1 output", activation.data, expected->input, QKV_ELEMENTS);
        } else {
            for (size_t i = 0; i < QKV_ELEMENTS; ++i) activation.data[i] = expected->input[i];
        }
        for (size_t i = 0; i < QKV_ELEMENTS; ++i) {
            q_output.data[i] = 93; k_output.data[i] = 93; v_output.data[i] = 93;
            accumulator.data[i] = INT32_C(1234567);
        }
        /* Real block orchestration: Q/K/V with one shared INT32 accumulator. */
        block->qkv(activation.data, q_output.data, k_output.data, v_output.data, accumulator.data);
        for (unsigned int p = 0; p < 3; ++p) {
            CHECK(projection_names[p], outputs[p], expected->output[p], QKV_ELEMENTS);
        }
        CHECK("final V accumulator", accumulator.data, expected->accumulator[2], QKV_ELEMENTS);
        if (check_guards(block_index, case_index)) return 1;

        /* Check each accumulator before its buffer is reused. */
        for (unsigned int p = 0; p < 3; ++p) {
            projection_accumulate_i8(activation.data, block->weights[p], NULL, accumulator.data,
                                     QKV_TOKENS, QKV_FEATURES, QKV_FEATURES, block->input_zero_point);
            CHECK(accumulator_names[p], accumulator.data, expected->accumulator[p], QKV_ELEMENTS);
            requantize_i32_to_i8(accumulator.data, outputs[p], QKV_ELEMENTS,
                                 block->multipliers[p], block->shift, block->output_zero_points[p]);
            CHECK(projection_names[p], outputs[p], expected->output[p], QKV_ELEMENTS);
            if (check_guards(block_index, case_index)) return 1;
        }
        CHECK("projection input unchanged", activation.data, expected->input, QKV_ELEMENTS);
        CHECK("block input unchanged", block_input.data, g->block_input, QKV_ELEMENTS);
    }
    return 0;
}

int main(void)
{
    for (unsigned int b = 0; b < 8; ++b) {
        for (unsigned int c = 0; c < QKV_CASE_COUNT; ++c) {
            if (test_case(b, c)) return 1;
        }
        printf("Block %u: %u standalone and LN1-chained QKV cases PASS\n", b, QKV_CASE_COUNT);
    }
    printf("PASS: 8 blocks, %u cases per path, %u Q/K/V outputs per path, "
           "0 mismatches; accumulators and guards checked\n",
           8u * QKV_CASE_COUNT, 8u * QKV_CASE_COUNT * 3u * QKV_ELEMENTS);
    return 0;
}
