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
#include "mlp_out_residual_golden.h"

typedef void (*Ln1Forward)(const int8_t *, int8_t *, LayerNormI8Workspace *);
typedef void (*QkvForward)(const int8_t *, int8_t *, int8_t *, int8_t *, int32_t *);
typedef void (*AttentionForward)(const int8_t *, const int8_t *, const int8_t *, int8_t *, int32_t *, int32_t *);
typedef void (*OutResidualForward)(const int8_t *, const int8_t *, int32_t *, int8_t *, int32_t *);
typedef void (*Ln2Forward)(const int32_t *, int8_t *, LayerNormI8Workspace *);
typedef void (*MlpForward)(const int8_t *, int8_t *, MlpGeluWorkspace *);
typedef void (*MlpOutForward)(const int8_t *, const int32_t *, int32_t *, int8_t *, int32_t *);
typedef void (*BoundaryForward)(const int32_t *, int8_t *);
typedef struct {
    Ln1Forward ln1;
    QkvForward qkv;
    AttentionForward attention;
    OutResidualForward out_residual;
    Ln2Forward ln2;
    MlpForward mlp_gelu;
    MlpOutForward mlp_out;
    BoundaryForward boundary;
    const int8_t *weights;
    const int32_t *bias;
    int32_t input_zero_point;
    int64_t multiplier;
    unsigned int shift;
    int32_t output_zero_point;
    int64_t branch_multiplier;
    unsigned int branch_shift;
    const MlpOutGolden *cases;
    const BoundaryGolden *probes;
} BlockEntry;
#define BLOCK_ENTRY(n) {block##n##_ln1_forward, block##n##_qkv_forward, \
    block##n##_attention_forward, block##n##_out_residual_forward, block##n##_ln2_forward, \
    block##n##_mlp_gelu_forward, block##n##_mlp_out_residual_forward, block##n##_output_boundary_forward, \
    block##n##_mlp_out_weights, block##n##_mlp_out_bias, BLOCK##n##_MLP_OUT_INPUT_ZERO_POINT, \
    BLOCK##n##_MLP_OUT_MULTIPLIER, BLOCK##n##_MLP_OUT_SHIFT, BLOCK##n##_MLP_OUT_OUTPUT_ZERO_POINT, \
    BLOCK##n##_RESIDUAL2_BRANCH_MULTIPLIER, BLOCK##n##_RESIDUAL2_BRANCH_SHIFT, \
    block##n##_mlp_out_residual_cases, &block##n##_boundary_probes}
static const BlockEntry blocks[] = {
    BLOCK_ENTRY(0), BLOCK_ENTRY(1), BLOCK_ENTRY(2), BLOCK_ENTRY(3),
    BLOCK_ENTRY(4), BLOCK_ENTRY(5), BLOCK_ENTRY(6), BLOCK_ENTRY(7)
};

#define GUARDED_TYPE(name, type, count) \
    typedef struct { uint64_t before; type data[count]; uint64_t after; } name
GUARDED_TYPE(GuardSmallI8, int8_t, MLP_OUT_ELEMENTS);
GUARDED_TYPE(GuardBigI8, int8_t, MLP_OUT_INPUT_ELEMENTS);
GUARDED_TYPE(GuardRow, int32_t, 32);
GUARDED_TYPE(GuardSmallI32, int32_t, MLP_OUT_ELEMENTS);
GUARDED_TYPE(GuardBigI32, int32_t, MLP_OUT_INPUT_ELEMENTS);
GUARDED_TYPE(GuardWide, int64_t, MLP_GELU_ADD_CHUNK_ELEMENTS);
GUARDED_TYPE(GuardScores, int32_t, MLP_OUT_SCORE_ELEMENTS);
static GuardSmallI8 block_input, activation, query, key, value, attention_output, projection_output, output;
static GuardSmallI32 retained, feature, output_i32;
static GuardRow mean_sum, variance, inv_std;
static GuardWide wide;
static GuardScores scores;
static GuardBigI8 expansion, half, branch, gelu_output;
static GuardBigI32 product;
static LayerNormI8Workspace ln_workspace = {
    mean_sum.data, variance.data, inv_std.data, feature.data, wide.data
};
static MlpGeluWorkspace mlp_workspace = {
    expansion.data, half.data, branch.data, product.data, wide.data
};
_Static_assert(MLP_GELU_ADD_CHUNK_ELEMENTS >= MLP_OUT_ELEMENTS, "LN fine workspace too small");
#define CANARY UINT64_C(0xb703e9a15c862df4)
#define SET_GUARD(buffer) do { (buffer).before = CANARY; (buffer).after = ~CANARY; } while (0)
#define CHECK_GUARD(buffer) \
    do { \
        if ((buffer).before != CANARY || (buffer).after != ~CANARY) { \
            printf("FAIL block %u case %u %s: %s guard changed\n", block_index, case_index, context, #buffer); \
            return 1; \
        } \
    } while (0)

static void reset_guards(void)
{
    SET_GUARD(block_input); SET_GUARD(activation); SET_GUARD(query); SET_GUARD(key);
    SET_GUARD(value); SET_GUARD(attention_output); SET_GUARD(projection_output); SET_GUARD(output);
    SET_GUARD(retained); SET_GUARD(feature); SET_GUARD(output_i32);
    SET_GUARD(mean_sum); SET_GUARD(variance); SET_GUARD(inv_std); SET_GUARD(wide); SET_GUARD(scores);
    SET_GUARD(expansion); SET_GUARD(half); SET_GUARD(branch); SET_GUARD(gelu_output); SET_GUARD(product);
}

static int check_guards(unsigned int block_index, unsigned int case_index, const char *context)
{
    CHECK_GUARD(block_input); CHECK_GUARD(activation); CHECK_GUARD(query); CHECK_GUARD(key);
    CHECK_GUARD(value); CHECK_GUARD(attention_output); CHECK_GUARD(projection_output); CHECK_GUARD(output);
    CHECK_GUARD(retained); CHECK_GUARD(feature); CHECK_GUARD(output_i32);
    CHECK_GUARD(mean_sum); CHECK_GUARD(variance); CHECK_GUARD(inv_std); CHECK_GUARD(wide); CHECK_GUARD(scores);
    CHECK_GUARD(expansion); CHECK_GUARD(half); CHECK_GUARD(branch); CHECK_GUARD(gelu_output); CHECK_GUARD(product);
    return 0;
}

#define CHECK(label, actual, expected, count) \
    do { \
        for (size_t check_i = 0; check_i < (size_t)(count); ++check_i) { \
            const int64_t got = (int64_t)(actual)[check_i], want = (int64_t)(expected)[check_i]; \
            if (got != want) { \
                printf("FAIL block %u case %u (%s), %s, %s[%u]: got %" PRId64 ", expected %" PRId64 "\n", \
                       block_index, case_index, mlp_out_residual_case_names[case_index], path_name, label, (unsigned int)check_i, got, want); \
                return 1; \
            } \
        } \
    } while (0)

static int test_boundary(unsigned int block_index)
{
    const BlockEntry *block = &blocks[block_index];
    const BoundaryGolden *g = block->probes;
    reset_guards();
    for (size_t i = 0; i < MLP_OUT_ELEMENTS; ++i) {
        output_i32.data[i] = g->input[i % BOUNDARY_PROBE_COUNT]; output.data[i] = 93;
    }
    block->boundary(output_i32.data, output.data);
    for (size_t i = 0; i < MLP_OUT_ELEMENTS; ++i) {
        const size_t p = i % BOUNDARY_PROBE_COUNT;
        if (output.data[i] != g->output[p] || output_i32.data[i] != g->input[p]) {
            printf("FAIL block %u boundary probe %u: got %d, expected %d, input=%" PRId32 "\n",
                   block_index, (unsigned int)p, (int)output.data[i], (int)g->output[p], output_i32.data[i]);
            return 1;
        }
    }
    return check_guards(block_index, 0, "boundary probes");
}

static int test_case(unsigned int block_index, unsigned int case_index)
{
    const BlockEntry *block = &blocks[block_index];
    const MlpOutGolden *g = &block->cases[case_index];
    const char *path_name = "setup";
    reset_guards();
    for (size_t i = 0; i < MLP_OUT_ELEMENTS; ++i) block_input.data[i] = g->block_input[i];
    for (unsigned int path = 0; path < 2; ++path) {
        path_name = path ? "Parts1-6-chained" : "standalone";
        const MlpOutPathGolden *expected = path ? &g->chained : &g->local;
        if (path) {
            block->ln1(block_input.data, activation.data, &ln_workspace);
            block->qkv(activation.data, query.data, key.data, value.data, feature.data);
            block->attention(query.data, key.data, value.data, attention_output.data, scores.data, feature.data);
            block->out_residual(attention_output.data, block_input.data, retained.data, projection_output.data, feature.data);
            block->ln2(retained.data, activation.data, &ln_workspace);
            block->mlp_gelu(activation.data, gelu_output.data, &mlp_workspace);
        } else {
            for (size_t i = 0; i < MLP_OUT_INPUT_ELEMENTS; ++i) gelu_output.data[i] = expected->input[i];
            for (size_t i = 0; i < MLP_OUT_ELEMENTS; ++i) retained.data[i] = expected->retained[i];
        }
        CHECK("GELU input", gelu_output.data, expected->input, MLP_OUT_INPUT_ELEMENTS);
        CHECK("retained Part-4 residual", retained.data, expected->retained, MLP_OUT_ELEMENTS);
        for (size_t i = 0; i < MLP_OUT_ELEMENTS; ++i) {
            output_i32.data[i] = -1234567; feature.data[i] = 1234567;
            projection_output.data[i] = 93; output.data[i] = 93;
        }

        /* Real Part-7 orchestration with distinct output and accumulator. */
        block->mlp_out(gelu_output.data, retained.data, output_i32.data, projection_output.data, feature.data);
        CHECK("forward INT8 projection", projection_output.data, expected->projection, MLP_OUT_ELEMENTS);
        CHECK("forward INT32 branch", feature.data, expected->branch, MLP_OUT_ELEMENTS);
        CHECK("forward INT32 sum", output_i32.data, expected->output_i32, MLP_OUT_ELEMENTS);
        block->boundary(output_i32.data, output.data);
        CHECK("forward final INT8", output.data, expected->output, MLP_OUT_ELEMENTS);
        if (check_guards(block_index, case_index, path_name)) return 1;

        /* Individual projection, conversion and residual checkpoints. */
        projection_accumulate_i8(gelu_output.data, block->weights, block->bias, feature.data,
                                 32, 512, 128, block->input_zero_point);
        CHECK("biased accumulator", feature.data, expected->accumulator, MLP_OUT_ELEMENTS);
        requantize_i32_to_i8(feature.data, projection_output.data, MLP_OUT_ELEMENTS,
                             block->multiplier, block->shift, block->output_zero_point);
        CHECK("INT8 projection", projection_output.data, expected->projection, MLP_OUT_ELEMENTS);
        requantize_i8_to_i32(projection_output.data, feature.data, MLP_OUT_ELEMENTS,
                             block->branch_multiplier, block->branch_shift, block->output_zero_point);
        CHECK("INT32 branch", feature.data, expected->branch, MLP_OUT_ELEMENTS);
        residual_add_i32(feature.data, retained.data, output_i32.data, MLP_OUT_ELEMENTS);
        CHECK("INT32 sum", output_i32.data, expected->output_i32, MLP_OUT_ELEMENTS);
        block->boundary(output_i32.data, output.data);
        CHECK("staged final INT8", output.data, expected->output, MLP_OUT_ELEMENTS);
        if (check_guards(block_index, case_index, path_name)) return 1;

        /* Permitted alias 1: INT32 output reuses the accumulator workspace. */
        block->mlp_out(gelu_output.data, retained.data, feature.data, projection_output.data, feature.data);
        CHECK("accumulator-alias sum", feature.data, expected->output_i32, MLP_OUT_ELEMENTS);
        CHECK("accumulator-alias projection", projection_output.data, expected->projection, MLP_OUT_ELEMENTS);
        block->boundary(feature.data, output.data);
        CHECK("accumulator-alias INT8", output.data, expected->output, MLP_OUT_ELEMENTS);
        CHECK("retained input preserved", retained.data, expected->retained, MLP_OUT_ELEMENTS);
        if (check_guards(block_index, case_index, path_name)) return 1;

        /* Permitted alias 2: output overwrites the retained residual after it is consumed. */
        block->mlp_out(gelu_output.data, retained.data, retained.data, projection_output.data, feature.data);
        CHECK("retained-alias sum", retained.data, expected->output_i32, MLP_OUT_ELEMENTS);
        block->boundary(retained.data, output.data);
        CHECK("retained-alias INT8", output.data, expected->output, MLP_OUT_ELEMENTS);
        CHECK("GELU input preserved", gelu_output.data, expected->input, MLP_OUT_INPUT_ELEMENTS);
        CHECK("original block input preserved", block_input.data, g->block_input, MLP_OUT_ELEMENTS);
        if (check_guards(block_index, case_index, path_name)) return 1;
    }
    return 0;
}

int main(void)
{
    for (unsigned int b = 0; b < 8; ++b) {
        if (test_boundary(b)) return 1;
        printf("Block %u: boundary rounding and saturation PASS\n", b);
        for (unsigned int c = 0; c < MLP_OUT_CASE_COUNT; ++c) if (test_case(b,c)) return 1;
        printf("Block %u: %u standalone and Parts1-6-chained MLP output/residual cases PASS\n", b, MLP_OUT_CASE_COUNT);
    }
    printf("PASS: 8 blocks, %u cases per path, %u final INT8 outputs per path, "
           "0 mismatches; INT32 checkpoints, boundary probes, buffer reuse and guards checked\n",
           8u*MLP_OUT_CASE_COUNT, 8u*MLP_OUT_CASE_COUNT*MLP_OUT_ELEMENTS);
    return 0;
}
