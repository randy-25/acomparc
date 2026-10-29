#include "block1.h"
#include "../requantization.h"
#include "../projection.h"
#include "../attention.h"
#include "../residual.h"
#include "../gelu.h"
#include "block1_params.h"

void block1_ln1_forward(const int8_t *input, int8_t *output,
                         LayerNormI8Workspace *workspace)
{
    const size_t tokens = BLOCK1_TOKEN_COUNT;
    const size_t features = BLOCK1_HIDDEN_SIZE;
    const size_t elements = tokens * features;
    const LayerNormI8Params *p = &block1_ln1_params;

    layernorm_sum_i8(input, workspace->mean_sum, tokens, features);
    layernorm_squared_difference_i8(input, workspace->mean_sum,
                                   workspace->fine_i64, tokens, features);

    /* Conversion 1: square INT64 -> INT32, XREQUANT mode 0. */
    requantize_shift_i64_to_i32(workspace->fine_i64, workspace->feature_i32,
                              elements, p->square_shift);
    layernorm_variance_i32(workspace->feature_i32, workspace->variance,
                          tokens, features);
    layernorm_add_epsilon_i32(workspace->variance, workspace->variance,
                             tokens, p->epsilon_q);
    layernorm_rsqrt_i32(workspace->variance, workspace->inv_std,
                       tokens, p->rsqrt_numerator);
    layernorm_normalize_i8(input, workspace->mean_sum, workspace->inv_std,
                          workspace->fine_i64, tokens, features);

    /* Conversion 2: normalized INT64 -> INT32 at 2^-16, mode 0. */
    requantize_shift_i64_to_i32(workspace->fine_i64, workspace->feature_i32,
                              elements, p->normalize_shift);
    layernorm_affine_i32(workspace->feature_i32, p->gamma, p->beta,
                        workspace->fine_i64, tokens, features,
                        p->beta_alignment);

    /* Conversion 3: affine INT64 at 2^-32 -> LN1 INT8, mode 1. */
    requantize_i64_to_i8(workspace->fine_i64, output, elements,
                        p->output_multiplier, p->output_shift,
                        p->output_zero_point);
}

void block1_qkv_forward(const int8_t *input,
                         int8_t *q_output, int8_t *k_output, int8_t *v_output,
                         int32_t *accumulator)
{
    const size_t tokens = BLOCK1_TOKEN_COUNT;
    const size_t features = BLOCK1_HIDDEN_SIZE;
    const size_t elements = tokens * features;

    /* Q: XTILE accumulation, then XREQUANT mode 2. No bias. */
    projection_accumulate_i8(input, block1_q_weights, NULL, accumulator,
                             tokens, features, features,
                             BLOCK1_QKV_INPUT_ZERO_POINT);
    requantize_i32_to_i8(accumulator, q_output, elements,
                         BLOCK1_Q_MULTIPLIER, BLOCK1_QKV_REQUANT_SHIFT,
                         BLOCK1_Q_OUTPUT_ZERO_POINT);

    /* K: XTILE accumulation, then XREQUANT mode 2. No bias. */
    projection_accumulate_i8(input, block1_k_weights, NULL, accumulator,
                             tokens, features, features,
                             BLOCK1_QKV_INPUT_ZERO_POINT);
    requantize_i32_to_i8(accumulator, k_output, elements,
                         BLOCK1_K_MULTIPLIER, BLOCK1_QKV_REQUANT_SHIFT,
                         BLOCK1_K_OUTPUT_ZERO_POINT);

    /* V: XTILE accumulation, then XREQUANT mode 2. No bias. */
    projection_accumulate_i8(input, block1_v_weights, NULL, accumulator,
                             tokens, features, features,
                             BLOCK1_QKV_INPUT_ZERO_POINT);
    requantize_i32_to_i8(accumulator, v_output, elements,
                         BLOCK1_V_MULTIPLIER, BLOCK1_QKV_REQUANT_SHIFT,
                         BLOCK1_V_OUTPUT_ZERO_POINT);

}

void block1_attention_scores_forward(const int8_t *query, const int8_t *key,
                                      int32_t *scores)
{
    attention_scores_i8(query, key, scores,
                         BLOCK1_TOKEN_COUNT, BLOCK1_HEAD_COUNT, BLOCK1_HEAD_DIM,
                         BLOCK1_Q_OUTPUT_ZERO_POINT, BLOCK1_K_OUTPUT_ZERO_POINT);
}

void block1_attention_forward(const int8_t *query, const int8_t *key,
                               const int8_t *value, int8_t *output,
                               int32_t *score_workspace, int32_t *context_workspace)
{
    /* Software Q*K^T in the existing token-major Q/K layout. */
    block1_attention_scores_forward(query, key, score_workspace);
    attention_causal_shift_i32(score_workspace, score_workspace,
                                BLOCK1_HEAD_COUNT, BLOCK1_TOKEN_COUNT);
    attention_exp_lut_i32(score_workspace, score_workspace,
                          BLOCK1_HEAD_COUNT, BLOCK1_TOKEN_COUNT,
                          block1_softmax_exp_lut,
                          BLOCK1_SOFTMAX_COORD_MULTIPLIER,
                          BLOCK1_SOFTMAX_COORD_SHIFT);
    attention_normalize_exp_i32(score_workspace, score_workspace,
                                BLOCK1_HEAD_COUNT, BLOCK1_TOKEN_COUNT,
                                BLOCK1_SOFTMAX_PROB_FRAC_BITS);

    /* Software P*V writes directly to token-major context. */
    attention_context_i8(score_workspace, value, context_workspace,
                          BLOCK1_TOKEN_COUNT, BLOCK1_HEAD_COUNT, BLOCK1_HEAD_DIM,
                          BLOCK1_V_OUTPUT_ZERO_POINT);

    /* Preserve the existing separate XREQUANT mode-2 context conversion. */
    requantize_i32_to_i8(context_workspace, output,
                         BLOCK1_TOKEN_COUNT * BLOCK1_HIDDEN_SIZE,
                         BLOCK1_ATTN_CONTEXT_MULTIPLIER,
                         BLOCK1_ATTN_CONTEXT_SHIFT,
                         BLOCK1_ATTN_CONTEXT_OUTPUT_ZERO_POINT);
}

void block1_out_residual_forward(const int8_t *attention_output,
                                  const int8_t *residual_input, int32_t *output,
                                  int8_t *projection_workspace,
                                  int32_t *accumulator_workspace)
{
    const size_t count = BLOCK1_TOKEN_COUNT * BLOCK1_HIDDEN_SIZE;

    /* XTILE computes the dot product; the shared kernel adds bias once. */
    projection_accumulate_i8(attention_output, block1_out_weights,
                             block1_out_bias, accumulator_workspace,
                             BLOCK1_TOKEN_COUNT, BLOCK1_HIDDEN_SIZE, BLOCK1_HIDDEN_SIZE,
                             BLOCK1_OUT_PROJ_INPUT_ZERO_POINT);

    /* Retain the saved INT8 projection boundary (XREQUANT mode 2). */
    requantize_i32_to_i8(accumulator_workspace, projection_workspace, count,
                         BLOCK1_OUT_PROJ_MULTIPLIER, BLOCK1_OUT_PROJ_SHIFT,
                         BLOCK1_OUT_PROJ_OUTPUT_ZERO_POINT);

    /* Convert that INT8 branch to INT32 at 2^-16 (XREQUANT mode 3). */
    requantize_i8_to_i32(projection_workspace, output, count,
                         BLOCK1_RESIDUAL1_BRANCH_MULTIPLIER,
                         BLOCK1_RESIDUAL1_BRANCH_SHIFT,
                         BLOCK1_OUT_PROJ_OUTPUT_ZERO_POINT);

    /* Align the original block input exactly, then add without saturation. */
    residual_add_i8_i32(residual_input, output, output, count,
                         BLOCK1_RESIDUAL1_INPUT_ALIGNMENT,
                         BLOCK1_RESIDUAL1_INPUT_ZERO_POINT);
}

void block1_ln2_forward(const int32_t *input, int8_t *output,
                         LayerNormI8Workspace *workspace)
{
    const size_t tokens = BLOCK1_TOKEN_COUNT;
    const size_t features = BLOCK1_HIDDEN_SIZE;
    const size_t elements = tokens * features;
    const LayerNormI8Params *p = &block1_ln2_params;

    /* Input scale 2^-16; feature-sum mean scale 2^-23. */
    layernorm_sum_i32(input, workspace->mean_sum, tokens, features);
    layernorm_squared_difference_i32(input, workspace->mean_sum,
                                     workspace->fine_i64, tokens, features);

    /* Conversion 1: square INT64 at 2^-46 -> INT32 at 2^-23, mode 0. */
    requantize_shift_i64_to_i32(workspace->fine_i64, workspace->feature_i32,
                              elements, p->square_shift);
    layernorm_variance_i32(workspace->feature_i32, workspace->variance,
                          tokens, features);
    layernorm_add_epsilon_i32(workspace->variance, workspace->variance,
                             tokens, p->epsilon_q);
    layernorm_rsqrt_i32(workspace->variance, workspace->inv_std,
                       tokens, p->rsqrt_numerator);
    layernorm_normalize_i32(input, workspace->mean_sum, workspace->inv_std,
                            workspace->fine_i64, tokens, features);

    /* Conversion 2: normalized INT64 at 2^-39 -> INT32 at 2^-16, mode 0. */
    requantize_shift_i64_to_i32(workspace->fine_i64, workspace->feature_i32,
                              elements, p->normalize_shift);
    layernorm_affine_i32(workspace->feature_i32, p->gamma, p->beta,
                        workspace->fine_i64, tokens, features,
                        p->beta_alignment);

    /* Conversion 3: affine INT64 at 2^-32 -> saved LN2 INT8 output, mode 1. */
    requantize_i64_to_i8(workspace->fine_i64, output, elements,
                        p->output_multiplier, p->output_shift,
                        p->output_zero_point);
}

void block1_mlp_expand_forward(
    const int8_t *input,
    int8_t *output,
    int32_t *accumulator
)
{
    projection_accumulate_i8(
        input,
        block1_mlp_expand_weights,
        block1_mlp_expand_bias,
        accumulator,
        BLOCK1_TOKEN_COUNT,
        BLOCK1_MLP_EXPAND_INPUT_FEATURES,
        BLOCK1_MLP_EXPAND_OUTPUT_FEATURES,
        BLOCK1_MLP_EXPAND_INPUT_ZERO_POINT
    );

    requantize_i32_to_i8(
        accumulator,
        output,
        BLOCK1_TOKEN_COUNT * BLOCK1_MLP_EXPAND_OUTPUT_FEATURES,
        BLOCK1_MLP_EXPAND_MULTIPLIER,
        BLOCK1_MLP_EXPAND_SHIFT,
        BLOCK1_MLP_EXPAND_OUTPUT_ZERO_POINT
    );
}

void block1_gelu_forward(const int8_t *input, int8_t *output,
                          MlpGeluWorkspace *workspace)
{
    const size_t count = BLOCK1_TOKEN_COUNT * BLOCK1_MLP_EXPAND_OUTPUT_FEATURES;

    /* Op 59-60: half-input branch -> T92. */
    gelu_multiply_scalar_i8(
        input, workspace->product_i32, count,
        BLOCK1_GELU_HALF_INPUT_ZERO_POINT,
        BLOCK1_GELU_HALF_CENTERED_CONSTANT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->half, count,
        BLOCK1_GELU_HALF_MULTIPLIER,
        BLOCK1_GELU_HALF_SHIFT,
        BLOCK1_GELU_HALF_OUTPUT_ZERO_POINT
    );

    /* Op 61-62: signed cube -> T94. */
    gelu_cube_i8(
        input, workspace->product_i32, count,
        BLOCK1_GELU_CUBE_INPUT_ZERO_POINT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK1_GELU_CUBE_MULTIPLIER,
        BLOCK1_GELU_CUBE_SHIFT,
        BLOCK1_GELU_CUBE_OUTPUT_ZERO_POINT
    );

    /* Op 63: weighted cubic term -> T96. */
    gelu_multiply_scalar_i8(
        workspace->branch, workspace->product_i32, count,
        BLOCK1_GELU_WEIGHTED_CUBE_INPUT_ZERO_POINT,
        BLOCK1_GELU_WEIGHTED_CUBE_CENTERED_CONSTANT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK1_GELU_WEIGHTED_CUBE_MULTIPLIER,
        BLOCK1_GELU_WEIGHTED_CUBE_SHIFT,
        BLOCK1_GELU_WEIGHTED_CUBE_OUTPUT_ZERO_POINT
    );

    /* Op 64: combine branches, rounding once, in chunks -> T97. */
    for (size_t offset = 0; offset < count;
         offset += MLP_GELU_ADD_CHUNK_ELEMENTS) {
        size_t chunk = count - offset;

        if (chunk > MLP_GELU_ADD_CHUNK_ELEMENTS) {
            chunk = MLP_GELU_ADD_CHUNK_ELEMENTS;
        }

        gelu_add_scaled_i8(
            input + offset,
            workspace->branch + offset,
            workspace->wide_i64,
            chunk,
            BLOCK1_GELU_ADD_LINEAR_ZERO_POINT,
            BLOCK1_GELU_ADD_CUBIC_ZERO_POINT,
            BLOCK1_GELU_ADD_LINEAR_MULTIPLIER,
            BLOCK1_GELU_ADD_CUBIC_MULTIPLIER
        );

        requantize_i64_to_i8(
            workspace->wide_i64,
            workspace->branch + offset,
            chunk,
            INT64_C(1),
            BLOCK1_GELU_ADD_SHIFT,
            BLOCK1_GELU_ADD_OUTPUT_ZERO_POINT
        );
    }

    /* Op 65: quantized tanh argument -> T99. */
    gelu_multiply_scalar_i8(
        workspace->branch, workspace->product_i32, count,
        BLOCK1_GELU_ARGUMENT_INPUT_ZERO_POINT,
        BLOCK1_GELU_ARGUMENT_CENTERED_CONSTANT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK1_GELU_ARGUMENT_MULTIPLIER,
        BLOCK1_GELU_ARGUMENT_SHIFT,
        BLOCK1_GELU_ARGUMENT_OUTPUT_ZERO_POINT
    );

    /* Op 66-67: tanh lookup -> T101. */
    gelu_tanh_lut_i8(
        workspace->branch,
        workspace->product_i32,
        count,
        block1_gelu_tanh_lut
    );

    /* Op 68: add integer one -> T102. */
    gelu_add_scalar_i32(
        workspace->product_i32,
        workspace->product_i32,
        count,
        BLOCK1_GELU_TANH_ONE
    );

    /* Op 69: quantized one-plus-tanh factor -> T103. */
    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK1_GELU_FACTOR_MULTIPLIER,
        BLOCK1_GELU_FACTOR_SHIFT,
        BLOCK1_GELU_FACTOR_OUTPUT_ZERO_POINT
    );

    /* Op 70: centered T92 * centered T103. */
    gelu_multiply_i8(
        workspace->half,
        workspace->branch,
        workspace->product_i32,
        count,
        BLOCK1_GELU_OUTPUT_HALF_ZERO_POINT,
        BLOCK1_GELU_OUTPUT_FACTOR_ZERO_POINT
    );

    /* Selected two-stage rounding -> T104. */
    requantize_two_round_i32_to_i8(
        workspace->product_i32,
        output,
        count,
        BLOCK1_GELU_OUTPUT_Q31_MULTIPLIER,
        BLOCK1_GELU_OUTPUT_SECOND_SHIFT,
        BLOCK1_GELU_OUTPUT_ZERO_POINT
    );
}

void block1_mlp_gelu_forward(const int8_t *input, int8_t *output,
                              MlpGeluWorkspace *workspace)
{
    block1_mlp_expand_forward(input, workspace->t89, workspace->product_i32);
    block1_gelu_forward(workspace->t89, output, workspace);
}

void block1_mlp_out_residual_forward(
    const int8_t *input,
    const int32_t *residual_input,
    int32_t *output,
    int8_t *projection_workspace,
    int32_t *accumulator_workspace
)
{
    const size_t count =
        BLOCK1_TOKEN_COUNT * BLOCK1_MLP_OUT_OUTPUT_FEATURES;

    /* Op 71: 512 -> 128 projection, including bias. */
    projection_accumulate_i8(
        input,
        block1_mlp_out_weights,
        block1_mlp_out_bias,
        accumulator_workspace,
        BLOCK1_TOKEN_COUNT,
        BLOCK1_MLP_OUT_INPUT_FEATURES,
        BLOCK1_MLP_OUT_OUTPUT_FEATURES,
        BLOCK1_MLP_OUT_INPUT_ZERO_POINT
    );

    /* Preserve the INT8 T105 boundary. */
    requantize_i32_to_i8(
        accumulator_workspace,
        projection_workspace,
        count,
        BLOCK1_MLP_OUT_MULTIPLIER,
        BLOCK1_MLP_OUT_SHIFT,
        BLOCK1_MLP_OUT_OUTPUT_ZERO_POINT
    );

    /* Op 72 counterpart: T105 -> INT32 T106, scale 2^-16. */
    requantize_i8_to_i32(
        projection_workspace,
        accumulator_workspace,
        count,
        BLOCK1_RESIDUAL2_BRANCH_MULTIPLIER,
        BLOCK1_RESIDUAL2_BRANCH_SHIFT,
        BLOCK1_MLP_OUT_OUTPUT_ZERO_POINT
    );

    /* Op 73: T106 + retained T73 -> T107. */
    residual_add_i32(
        accumulator_workspace,
        residual_input,
        output,
        count
    );
}

void block1_output_boundary_forward(const int32_t *input, int8_t *output)
{
    requantize_i32_to_i8(input, output,
                         BLOCK1_TOKEN_COUNT * BLOCK1_HIDDEN_SIZE,
                         BLOCK1_OUTPUT_REQUANT_MULTIPLIER,
                         BLOCK1_OUTPUT_REQUANT_SHIFT,
                         BLOCK1_OUTPUT_REQUANT_ZERO_POINT);
}

void block1_forward(const int8_t *input, int8_t *output,
                      TransformerBlockWorkspace *workspace)
{
    block1_ln1_forward(input, workspace->activation, &workspace->ln);
    block1_qkv_forward(workspace->activation, workspace->query, workspace->key,
                         workspace->value, workspace->ln.feature_i32);
    block1_attention_forward(workspace->query, workspace->key, workspace->value,
                               workspace->attention_output, workspace->score_workspace,
                               workspace->ln.feature_i32);

    /* Attention scores are dead; retain Part 4's residual in their buffer. */
    block1_out_residual_forward(workspace->attention_output, input,
                                  workspace->score_workspace, workspace->projection_output,
                                  workspace->ln.feature_i32);
    block1_ln2_forward(workspace->score_workspace, workspace->activation, &workspace->ln);
    block1_mlp_gelu_forward(workspace->activation, workspace->mlp_output, &workspace->mlp);

    /* Part 7 consumes the retained residual and replaces it with the final sum. */
    block1_mlp_out_residual_forward(workspace->mlp_output, workspace->score_workspace,
                                      workspace->score_workspace, workspace->projection_output,
                                      workspace->ln.feature_i32);
    block1_output_boundary_forward(workspace->score_workspace, output);
}
