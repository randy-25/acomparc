#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "gelu.h"
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
#include "mlp_gelu_golden.h"

typedef void (*Ln1Forward)(const int8_t *, int8_t *, LayerNormI8Workspace *);
typedef void (*QkvForward)(const int8_t *, int8_t *, int8_t *, int8_t *, int32_t *);
typedef void (*AttentionForward)(const int8_t *, const int8_t *, const int8_t *, int8_t *, int32_t *, int32_t *);
typedef void (*OutResidualForward)(const int8_t *, const int8_t *, int32_t *, int8_t *, int32_t *);
typedef void (*Ln2Forward)(const int32_t *, int8_t *, LayerNormI8Workspace *);
typedef void (*ExpandForward)(const int8_t *, int8_t *, int32_t *);
typedef void (*MlpForward)(const int8_t *, int8_t *, MlpGeluWorkspace *);
typedef struct { int64_t multiplier; unsigned int shift; int32_t zero_point; } Conversion;
typedef struct { int32_t input_zero_point, scalar; Conversion conversion; } Scalar;
typedef struct { int32_t input_zero_point; Conversion conversion; } Cube;
typedef struct {
    int32_t linear_zero_point, cubic_zero_point;
    int64_t linear_multiplier, cubic_multiplier;
    unsigned int shift;
    int32_t zero_point;
} BranchSum;
typedef struct {
    Scalar half;
    Cube cube;
    Scalar weighted, argument;
    BranchSum sum;
    const int32_t *tanh_table;
    Conversion factor;
    int32_t half_zero_point, factor_zero_point, q31_multiplier;
    unsigned int second_shift;
    int32_t output_zero_point;
} GeluSpec;
typedef struct {
    Ln1Forward ln1;
    QkvForward qkv;
    AttentionForward attention;
    OutResidualForward out_residual;
    Ln2Forward ln2;
    ExpandForward expand;
    MlpForward gelu_only, mlp_gelu;
    GeluSpec gelu;
    const MlpGeluGolden *cases;
    const GeluGolden *codes;
} BlockEntry;

#define CONVERSION(n,tag) {BLOCK##n##_GELU_##tag##_MULTIPLIER, \
    BLOCK##n##_GELU_##tag##_SHIFT, BLOCK##n##_GELU_##tag##_OUTPUT_ZERO_POINT}
#define SCALAR(n,tag) {BLOCK##n##_GELU_##tag##_INPUT_ZERO_POINT, \
    BLOCK##n##_GELU_##tag##_CENTERED_CONSTANT, CONVERSION(n,tag)}
#define GELU_SPEC(n) {SCALAR(n,HALF), \
    {BLOCK##n##_GELU_CUBE_INPUT_ZERO_POINT, CONVERSION(n,CUBE)}, \
    SCALAR(n,WEIGHTED_CUBE), SCALAR(n,ARGUMENT), \
    {BLOCK##n##_GELU_ADD_LINEAR_ZERO_POINT, BLOCK##n##_GELU_ADD_CUBIC_ZERO_POINT, \
     BLOCK##n##_GELU_ADD_LINEAR_MULTIPLIER, BLOCK##n##_GELU_ADD_CUBIC_MULTIPLIER, \
     BLOCK##n##_GELU_ADD_SHIFT, BLOCK##n##_GELU_ADD_OUTPUT_ZERO_POINT}, \
    block##n##_gelu_tanh_lut, CONVERSION(n,FACTOR), \
    BLOCK##n##_GELU_OUTPUT_HALF_ZERO_POINT, BLOCK##n##_GELU_OUTPUT_FACTOR_ZERO_POINT, \
    BLOCK##n##_GELU_OUTPUT_Q31_MULTIPLIER, BLOCK##n##_GELU_OUTPUT_SECOND_SHIFT, BLOCK##n##_GELU_OUTPUT_ZERO_POINT}
#define BLOCK_ENTRY(n) {block##n##_ln1_forward, block##n##_qkv_forward, \
    block##n##_attention_forward, block##n##_out_residual_forward, block##n##_ln2_forward, \
    block##n##_mlp_expand_forward, block##n##_gelu_forward, block##n##_mlp_gelu_forward, \
    GELU_SPEC(n), block##n##_mlp_gelu_cases, &block##n##_gelu_codes}
static const BlockEntry blocks[] = {
    BLOCK_ENTRY(0), BLOCK_ENTRY(1), BLOCK_ENTRY(2), BLOCK_ENTRY(3),
    BLOCK_ENTRY(4), BLOCK_ENTRY(5), BLOCK_ENTRY(6), BLOCK_ENTRY(7)
};

#define GUARDED_TYPE(name, type, count) \
    typedef struct { uint64_t before; type data[count]; uint64_t after; } name
GUARDED_TYPE(GuardSmallI8, int8_t, MLP_INPUT_ELEMENTS);
GUARDED_TYPE(GuardBigI8, int8_t, MLP_OUTPUT_ELEMENTS);
GUARDED_TYPE(GuardRow, int32_t, 32);
GUARDED_TYPE(GuardSmallI32, int32_t, MLP_INPUT_ELEMENTS);
GUARDED_TYPE(GuardBigI32, int32_t, MLP_OUTPUT_ELEMENTS);
GUARDED_TYPE(GuardWide, int64_t, MLP_GELU_ADD_CHUNK_ELEMENTS);
GUARDED_TYPE(GuardScores, int32_t, MLP_SCORE_ELEMENTS);
static GuardSmallI8 block_input, activation, query, key, value, attention_output, projection_output;
static GuardSmallI32 residual, feature;
static GuardRow mean_sum, variance, inv_std;
static GuardWide wide;
static GuardScores scores;
static GuardBigI8 expansion, half, branch, output;
static GuardBigI32 product;
static LayerNormI8Workspace ln_workspace = {
    mean_sum.data, variance.data, inv_std.data, feature.data, wide.data
};
static MlpGeluWorkspace mlp_workspace = {
    expansion.data, half.data, branch.data, product.data, wide.data
};
_Static_assert(MLP_GELU_ADD_CHUNK_ELEMENTS >= MLP_INPUT_ELEMENTS, "LN fine workspace too small");
#define CANARY UINT64_C(0xe36108b45c927dfa)
#define SET_GUARD(buffer) do { (buffer).before = CANARY; (buffer).after = ~CANARY; } while (0)
#define CHECK_GUARD(buffer) \
    do { \
        if ((buffer).before != CANARY || (buffer).after != ~CANARY) { \
            printf("FAIL block %u %s: %s guard changed\n", block_index, context, #buffer); \
            return 1; \
        } \
    } while (0)

static void reset_guards(void)
{
    SET_GUARD(block_input); SET_GUARD(activation); SET_GUARD(query); SET_GUARD(key);
    SET_GUARD(value); SET_GUARD(attention_output); SET_GUARD(projection_output);
    SET_GUARD(residual); SET_GUARD(feature); SET_GUARD(mean_sum); SET_GUARD(variance);
    SET_GUARD(inv_std); SET_GUARD(wide); SET_GUARD(scores);
    SET_GUARD(expansion); SET_GUARD(half); SET_GUARD(branch); SET_GUARD(output); SET_GUARD(product);
}

static int check_guards(unsigned int block_index, const char *context)
{
    CHECK_GUARD(block_input); CHECK_GUARD(activation); CHECK_GUARD(query); CHECK_GUARD(key);
    CHECK_GUARD(value); CHECK_GUARD(attention_output); CHECK_GUARD(projection_output);
    CHECK_GUARD(residual); CHECK_GUARD(feature); CHECK_GUARD(mean_sum); CHECK_GUARD(variance);
    CHECK_GUARD(inv_std); CHECK_GUARD(wide); CHECK_GUARD(scores);
    CHECK_GUARD(expansion); CHECK_GUARD(half); CHECK_GUARD(branch); CHECK_GUARD(output); CHECK_GUARD(product);
    return 0;
}

#define CHECK(label, actual, expected, count) \
    do { \
        for (size_t check_i = 0; check_i < (size_t)(count); ++check_i) { \
            const int64_t got = (int64_t)(actual)[check_i], want = (int64_t)(expected)[check_i]; \
            if (got != want) { \
                printf("FAIL block %u case %u (%s), %s, %s[%u]: got %" PRId64 ", expected %" PRId64 "\n", \
                       block_index, case_index, mlp_gelu_case_names[case_index], path_name, label, (unsigned int)check_i, got, want); \
                return 1; \
            } \
        } \
    } while (0)

/* Exhaustive GELU fixtures have period 256 throughout the 16384-element buffer. */
#define CHECK_CODES(label, actual, expected, count, offset) \
    do { \
        for (size_t check_i = 0; check_i < (size_t)(count); ++check_i) { \
            const int64_t got = (int64_t)(actual)[check_i]; \
            const int64_t want = (int64_t)(expected)[((size_t)(offset) + check_i) % GELU_CODE_COUNT]; \
            if (got != want) { \
                printf("FAIL block %u exhaustive GELU %s[%u]: got %" PRId64 ", expected %" PRId64 "\n", \
                       block_index, label, (unsigned int)((size_t)(offset)+check_i), got, want); \
                return 1; \
            } \
        } \
    } while (0)

static void convert_i32(const int32_t *src, int8_t *dst, const Conversion *p)
{
    requantize_i32_to_i8(src, dst, MLP_OUTPUT_ELEMENTS, p->multiplier, p->shift, p->zero_point);
}

static int test_gelu_codes(unsigned int block_index)
{
    const BlockEntry *block = &blocks[block_index];
    const GeluSpec *p = &block->gelu;
    const GeluGolden *g = block->codes;
    const size_t count = MLP_OUTPUT_ELEMENTS;
    reset_guards();
    for (size_t i = 0; i < count; ++i) {
        expansion.data[i] = (int8_t)((int)(i % GELU_CODE_COUNT) - 128);
        half.data[i] = 93; branch.data[i] = 93; product.data[i] = 1234567; output.data[i] = 93;
    }
    block->gelu_only(expansion.data, output.data, &mlp_workspace);
    CHECK_CODES("forward output", output.data, g->output, count, 0);
    CHECK_CODES("forward half", half.data, g->half, count, 0);
    CHECK_CODES("forward factor", branch.data, g->factor, count, 0);
    CHECK_CODES("forward product", product.data, g->product, count, 0);
    if (check_guards(block_index, "exhaustive forward")) return 1;

    gelu_multiply_scalar_i8(expansion.data, product.data, count, p->half.input_zero_point, p->half.scalar);
    CHECK_CODES("half raw", product.data, g->half_raw, count, 0);
    convert_i32(product.data, half.data, &p->half.conversion);
    CHECK_CODES("half", half.data, g->half, count, 0);
    gelu_cube_i8(expansion.data, product.data, count, p->cube.input_zero_point);
    CHECK_CODES("cube raw", product.data, g->cube_raw, count, 0);
    convert_i32(product.data, branch.data, &p->cube.conversion);
    CHECK_CODES("cube", branch.data, g->cube, count, 0);
    gelu_multiply_scalar_i8(branch.data, product.data, count, p->weighted.input_zero_point, p->weighted.scalar);
    CHECK_CODES("weighted raw", product.data, g->weighted_raw, count, 0);
    convert_i32(product.data, branch.data, &p->weighted.conversion);
    CHECK_CODES("weighted cube", branch.data, g->weighted_cube, count, 0);
    for (size_t offset = 0; offset < count; offset += MLP_GELU_ADD_CHUNK_ELEMENTS) {
        size_t chunk = count-offset;
        if (chunk > MLP_GELU_ADD_CHUNK_ELEMENTS) chunk = MLP_GELU_ADD_CHUNK_ELEMENTS;
        gelu_add_scaled_i8(expansion.data+offset, branch.data+offset, wide.data, chunk,
                           p->sum.linear_zero_point, p->sum.cubic_zero_point,
                           p->sum.linear_multiplier, p->sum.cubic_multiplier);
        CHECK_CODES("combined wide", wide.data, g->combined, chunk, offset);
        requantize_i64_to_i8(wide.data, branch.data+offset, chunk, INT64_C(1), p->sum.shift, p->sum.zero_point);
    }
    CHECK_CODES("sum", branch.data, g->sum, count, 0);
    gelu_multiply_scalar_i8(branch.data, product.data, count, p->argument.input_zero_point, p->argument.scalar);
    CHECK_CODES("argument raw", product.data, g->argument_raw, count, 0);
    convert_i32(product.data, branch.data, &p->argument.conversion);
    CHECK_CODES("argument", branch.data, g->argument, count, 0);
    gelu_tanh_lut_i8(branch.data, product.data, count, p->tanh_table);
    CHECK_CODES("tanh", product.data, g->tanh, count, 0);
    gelu_add_scalar_i32(product.data, product.data, count, INT32_C(16777216));
    CHECK_CODES("factor fine", product.data, g->factor_fine, count, 0);
    convert_i32(product.data, branch.data, &p->factor);
    CHECK_CODES("factor", branch.data, g->factor, count, 0);
    gelu_multiply_i8(half.data, branch.data, product.data, count, p->half_zero_point, p->factor_zero_point);
    CHECK_CODES("product", product.data, g->product, count, 0);
    requantize_two_round_i32_to_i8(product.data, output.data, count,
                                  p->q31_multiplier, p->second_shift, p->output_zero_point);
    CHECK_CODES("staged output", output.data, g->output, count, 0);
    for (size_t i = 0; i < count; ++i) {
        if (expansion.data[i] != (int8_t)((int)(i % GELU_CODE_COUNT)-128)) {
            printf("FAIL block %u: exhaustive GELU input changed\n", block_index); return 1;
        }
    }
    return check_guards(block_index, "exhaustive staged");
}

static int test_case(unsigned int block_index, unsigned int case_index)
{
    const BlockEntry *block = &blocks[block_index];
    const MlpGeluGolden *g = &block->cases[case_index];
    const char *path_name = "setup";
    reset_guards();
    for (size_t i = 0; i < MLP_INPUT_ELEMENTS; ++i) block_input.data[i] = g->block_input[i];
    for (unsigned int path = 0; path < 2; ++path) {
        path_name = path ? "Parts1-5-chained" : "standalone";
        const MlpGeluPathGolden *expected = path ? &g->chained : &g->local;
        if (path) {
            block->ln1(block_input.data, activation.data, &ln_workspace);
            block->qkv(activation.data, query.data, key.data, value.data, feature.data);
            block->attention(query.data, key.data, value.data, attention_output.data, scores.data, feature.data);
            block->out_residual(attention_output.data, block_input.data, residual.data, projection_output.data, feature.data);
            block->ln2(residual.data, activation.data, &ln_workspace);
        } else {
            for (size_t i = 0; i < MLP_INPUT_ELEMENTS; ++i) activation.data[i] = expected->input[i];
        }
        CHECK("LN2/expansion input", activation.data, expected->input, MLP_INPUT_ELEMENTS);
        for (size_t i = 0; i < MLP_OUTPUT_ELEMENTS; ++i) {
            expansion.data[i] = 93; half.data[i] = 93; branch.data[i] = 93;
            output.data[i] = 93; product.data[i] = 1234567;
        }
        block->mlp_gelu(activation.data, output.data, &mlp_workspace);
        CHECK("forward expansion", expansion.data, expected->expansion, MLP_OUTPUT_ELEMENTS);
        CHECK("forward GELU", output.data, expected->output, MLP_OUTPUT_ELEMENTS);
        if (check_guards(block_index, path_name)) return 1;

        block->expand(activation.data, expansion.data, product.data);
        CHECK("biased expansion accumulator", product.data, expected->accumulator, MLP_OUTPUT_ELEMENTS);
        CHECK("INT8 expansion", expansion.data, expected->expansion, MLP_OUTPUT_ELEMENTS);
        block->gelu_only(expansion.data, output.data, &mlp_workspace);
        CHECK("separate GELU", output.data, expected->output, MLP_OUTPUT_ELEMENTS);
        CHECK("expansion preserved", expansion.data, expected->expansion, MLP_OUTPUT_ELEMENTS);
        CHECK("LN2 input preserved", activation.data, expected->input, MLP_INPUT_ELEMENTS);
        CHECK("block input preserved", block_input.data, g->block_input, MLP_INPUT_ELEMENTS);
        if (check_guards(block_index, path_name)) return 1;
    }
    return 0;
}

int main(void)
{
    for (unsigned int b = 0; b < 8; ++b) {
        if (test_gelu_codes(b)) return 1;
        printf("Block %u: all 256 GELU input codes and checkpoints PASS\n", b);
        for (unsigned int c = 0; c < MLP_GELU_CASE_COUNT; ++c) if (test_case(b,c)) return 1;
        printf("Block %u: %u standalone and Parts1-5-chained MLP/GELU cases PASS\n", b, MLP_GELU_CASE_COUNT);
    }
    printf("PASS: 8 blocks, %u cases per path, %u MLP/GELU INT8 outputs per path, "
           "0 mismatches; all 256 GELU codes per block and guards checked\n",
           8u*MLP_GELU_CASE_COUNT, 8u*MLP_GELU_CASE_COUNT*MLP_OUTPUT_ELEMENTS);
    return 0;
}
