#include "block5.h"
#include "../requantization.h"
#include "../projection.h"
#include "../attention.h"
#include "../residual.h"
#include "../gelu.h"
#include "block5_params.h"

void block5_ln1_forward(const int8_t *input, int8_t *output,
                         LayerNormI8Workspace *workspace)
{
    const size_t tokens = BLOCK5_TOKEN_COUNT;
    const size_t features = BLOCK5_HIDDEN_SIZE;
    const size_t elements = tokens * features;
    const LayerNormI8Params *p = &block5_ln1_params;

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

void block5_qkv_forward(const int8_t *input,
                         int8_t *q_output, int8_t *k_output, int8_t *v_output,
                         int32_t *accumulator)
{
    const size_t tokens = BLOCK5_TOKEN_COUNT;
    const size_t features = BLOCK5_HIDDEN_SIZE;
    const size_t elements = tokens * features;

    /* Q: XTILE accumulation, then XREQUANT mode 2. No bias. */
    projection_accumulate_i8(input, block5_q_weights, NULL, accumulator,
                             tokens, features, features,
                             BLOCK5_QKV_INPUT_ZERO_POINT);
    requantize_i32_to_i8(accumulator, q_output, elements,
                         BLOCK5_Q_MULTIPLIER, BLOCK5_QKV_REQUANT_SHIFT,
                         BLOCK5_Q_OUTPUT_ZERO_POINT);

    /* K: XTILE accumulation, then XREQUANT mode 2. No bias. */
    projection_accumulate_i8(input, block5_k_weights, NULL, accumulator,
                             tokens, features, features,
                             BLOCK5_QKV_INPUT_ZERO_POINT);
    requantize_i32_to_i8(accumulator, k_output, elements,
                         BLOCK5_K_MULTIPLIER, BLOCK5_QKV_REQUANT_SHIFT,
                         BLOCK5_K_OUTPUT_ZERO_POINT);

    /* V: XTILE accumulation, then XREQUANT mode 2. No bias. */
    projection_accumulate_i8(input, block5_v_weights, NULL, accumulator,
                             tokens, features, features,
                             BLOCK5_QKV_INPUT_ZERO_POINT);
    requantize_i32_to_i8(accumulator, v_output, elements,
                         BLOCK5_V_MULTIPLIER, BLOCK5_QKV_REQUANT_SHIFT,
                         BLOCK5_V_OUTPUT_ZERO_POINT);

}

void block5_attention_scores_forward(const int8_t *query, const int8_t *key,
                                      int32_t *scores)
{
    attention_scores_i8(query, key, scores,
                         BLOCK5_TOKEN_COUNT, BLOCK5_HEAD_COUNT, BLOCK5_HEAD_DIM,
                         BLOCK5_Q_OUTPUT_ZERO_POINT, BLOCK5_K_OUTPUT_ZERO_POINT);
}

void block5_attention_forward(const int8_t *query, const int8_t *key,
                               const int8_t *value, int8_t *output,
                               int32_t *score_workspace, int32_t *context_workspace)
{
    /* Software Q*K^T in the existing token-major Q/K layout. */
    block5_attention_scores_forward(query, key, score_workspace);
    attention_causal_shift_i32(score_workspace, score_workspace,
                                BLOCK5_HEAD_COUNT, BLOCK5_TOKEN_COUNT);
    attention_exp_lut_i32(score_workspace, score_workspace,
                          BLOCK5_HEAD_COUNT, BLOCK5_TOKEN_COUNT,
                          block5_softmax_exp_lut,
                          BLOCK5_SOFTMAX_COORD_MULTIPLIER,
                          BLOCK5_SOFTMAX_COORD_SHIFT);
    attention_normalize_exp_i32(score_workspace, score_workspace,
                                BLOCK5_HEAD_COUNT, BLOCK5_TOKEN_COUNT,
                                BLOCK5_SOFTMAX_PROB_FRAC_BITS);

    /* Software P*V writes directly to token-major context. */
    attention_context_i8(score_workspace, value, context_workspace,
                          BLOCK5_TOKEN_COUNT, BLOCK5_HEAD_COUNT, BLOCK5_HEAD_DIM,
                          BLOCK5_V_OUTPUT_ZERO_POINT);

    /* Preserve the existing separate XREQUANT mode-2 context conversion. */
    requantize_i32_to_i8(context_workspace, output,
                         BLOCK5_TOKEN_COUNT * BLOCK5_HIDDEN_SIZE,
                         BLOCK5_ATTN_CONTEXT_MULTIPLIER,
                         BLOCK5_ATTN_CONTEXT_SHIFT,
                         BLOCK5_ATTN_CONTEXT_OUTPUT_ZERO_POINT);
}

void block5_out_residual_forward(const int8_t *attention_output,
                                  const int8_t *residual_input, int32_t *output,
                                  int8_t *projection_workspace,
                                  int32_t *accumulator_workspace)
{
    const size_t count = BLOCK5_TOKEN_COUNT * BLOCK5_HIDDEN_SIZE;

    /* XTILE computes the dot product; the shared kernel adds bias once. */
    projection_accumulate_i8(attention_output, block5_out_weights,
                             block5_out_bias, accumulator_workspace,
                             BLOCK5_TOKEN_COUNT, BLOCK5_HIDDEN_SIZE, BLOCK5_HIDDEN_SIZE,
                             BLOCK5_OUT_PROJ_INPUT_ZERO_POINT);

    /* Retain the saved INT8 projection boundary (XREQUANT mode 2). */
    requantize_i32_to_i8(accumulator_workspace, projection_workspace, count,
                         BLOCK5_OUT_PROJ_MULTIPLIER, BLOCK5_OUT_PROJ_SHIFT,
                         BLOCK5_OUT_PROJ_OUTPUT_ZERO_POINT);

    /* Convert that INT8 branch to INT32 at 2^-16 (XREQUANT mode 3). */
    requantize_i8_to_i32(projection_workspace, output, count,
                         BLOCK5_RESIDUAL1_BRANCH_MULTIPLIER,
                         BLOCK5_RESIDUAL1_BRANCH_SHIFT,
                         BLOCK5_OUT_PROJ_OUTPUT_ZERO_POINT);

    /* Align the original block input exactly, then add without saturation. */
    residual_add_i8_i32(residual_input, output, output, count,
                         BLOCK5_RESIDUAL1_INPUT_ALIGNMENT,
                         BLOCK5_RESIDUAL1_INPUT_ZERO_POINT);
}

void block5_ln2_forward(const int32_t *input, int8_t *output,
                         LayerNormI8Workspace *workspace)
{
    const size_t tokens = BLOCK5_TOKEN_COUNT;
    const size_t features = BLOCK5_HIDDEN_SIZE;
    const size_t elements = tokens * features;
    const LayerNormI8Params *p = &block5_ln2_params;

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

void block5_mlp_expand_forward(
    const int8_t *input,
    int8_t *output,
    int32_t *accumulator
)
{
    projection_accumulate_i8(
        input,
        block5_mlp_expand_weights,
        block5_mlp_expand_bias,
        accumulator,
        BLOCK5_TOKEN_COUNT,
        BLOCK5_MLP_EXPAND_INPUT_FEATURES,
        BLOCK5_MLP_EXPAND_OUTPUT_FEATURES,
        BLOCK5_MLP_EXPAND_INPUT_ZERO_POINT
    );

    requantize_i32_to_i8(
        accumulator,
        output,
        BLOCK5_TOKEN_COUNT * BLOCK5_MLP_EXPAND_OUTPUT_FEATURES,
        BLOCK5_MLP_EXPAND_MULTIPLIER,
        BLOCK5_MLP_EXPAND_SHIFT,
        BLOCK5_MLP_EXPAND_OUTPUT_ZERO_POINT
    );
}

void block5_gelu_forward(const int8_t *input, int8_t *output,
                          MlpGeluWorkspace *workspace)
{
    const size_t count = BLOCK5_TOKEN_COUNT * BLOCK5_MLP_EXPAND_OUTPUT_FEATURES;

    /* Op 59-60: half-input branch -> T92. */
    gelu_multiply_scalar_i8(
        input, workspace->product_i32, count,
        BLOCK5_GELU_HALF_INPUT_ZERO_POINT,
        BLOCK5_GELU_HALF_CENTERED_CONSTANT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->half, count,
        BLOCK5_GELU_HALF_MULTIPLIER,
        BLOCK5_GELU_HALF_SHIFT,
        BLOCK5_GELU_HALF_OUTPUT_ZERO_POINT
    );

    /* Op 61-62: signed cube -> T94. */
    gelu_cube_i8(
        input, workspace->product_i32, count,
        BLOCK5_GELU_CUBE_INPUT_ZERO_POINT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK5_GELU_CUBE_MULTIPLIER,
        BLOCK5_GELU_CUBE_SHIFT,
        BLOCK5_GELU_CUBE_OUTPUT_ZERO_POINT
    );

    /* Op 63: weighted cubic term -> T96. */
    gelu_multiply_scalar_i8(
        workspace->branch, workspace->product_i32, count,
        BLOCK5_GELU_WEIGHTED_CUBE_INPUT_ZERO_POINT,
        BLOCK5_GELU_WEIGHTED_CUBE_CENTERED_CONSTANT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK5_GELU_WEIGHTED_CUBE_MULTIPLIER,
        BLOCK5_GELU_WEIGHTED_CUBE_SHIFT,
        BLOCK5_GELU_WEIGHTED_CUBE_OUTPUT_ZERO_POINT
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
            BLOCK5_GELU_ADD_LINEAR_ZERO_POINT,
            BLOCK5_GELU_ADD_CUBIC_ZERO_POINT,
            BLOCK5_GELU_ADD_LINEAR_MULTIPLIER,
            BLOCK5_GELU_ADD_CUBIC_MULTIPLIER
        );

        requantize_i64_to_i8(
            workspace->wide_i64,
            workspace->branch + offset,
            chunk,
            INT64_C(1),
            BLOCK5_GELU_ADD_SHIFT,
            BLOCK5_GELU_ADD_OUTPUT_ZERO_POINT
        );
    }

    /* Op 65: quantized tanh argument -> T99. */
    gelu_multiply_scalar_i8(
        workspace->branch, workspace->product_i32, count,
        BLOCK5_GELU_ARGUMENT_INPUT_ZERO_POINT,
        BLOCK5_GELU_ARGUMENT_CENTERED_CONSTANT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK5_GELU_ARGUMENT_MULTIPLIER,
        BLOCK5_GELU_ARGUMENT_SHIFT,
        BLOCK5_GELU_ARGUMENT_OUTPUT_ZERO_POINT
    );

    /* Op 66-67: tanh lookup -> T101. */
    gelu_tanh_lut_i8(
        workspace->branch,
        workspace->product_i32,
        count,
        block5_gelu_tanh_lut
    );

    /* Op 68: add integer one -> T102. */
    gelu_add_scalar_i32(
        workspace->product_i32,
        workspace->product_i32,
        count,
        BLOCK5_GELU_TANH_ONE
    );

    /* Op 69: quantized one-plus-tanh factor -> T103. */
    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK5_GELU_FACTOR_MULTIPLIER,
        BLOCK5_GELU_FACTOR_SHIFT,
        BLOCK5_GELU_FACTOR_OUTPUT_ZERO_POINT
    );

    /* Op 70: centered T92 * centered T103. */
    gelu_multiply_i8(
        workspace->half,
        workspace->branch,
        workspace->product_i32,
        count,
        BLOCK5_GELU_OUTPUT_HALF_ZERO_POINT,
        BLOCK5_GELU_OUTPUT_FACTOR_ZERO_POINT
    );

    /* Selected two-stage rounding -> T104. */
    requantize_two_round_i32_to_i8(
        workspace->product_i32,
        output,
        count,
        BLOCK5_GELU_OUTPUT_Q31_MULTIPLIER,
        BLOCK5_GELU_OUTPUT_SECOND_SHIFT,
        BLOCK5_GELU_OUTPUT_ZERO_POINT
    );
}

void block5_mlp_gelu_forward(const int8_t *input, int8_t *output,
                              MlpGeluWorkspace *workspace)
{
    block5_mlp_expand_forward(input, workspace->t89, workspace->product_i32);
    block5_gelu_forward(workspace->t89, output, workspace);
}

void block5_mlp_out_residual_forward(
    const int8_t *input,
    const int32_t *residual_input,
    int32_t *output,
    int8_t *projection_workspace,
    int32_t *accumulator_workspace
)
{
    const size_t count =
        BLOCK5_TOKEN_COUNT * BLOCK5_MLP_OUT_OUTPUT_FEATURES;

    /* Op 71: 512 -> 128 projection, including bias. */
    projection_accumulate_i8(
        input,
        block5_mlp_out_weights,
        block5_mlp_out_bias,
        accumulator_workspace,
        BLOCK5_TOKEN_COUNT,
        BLOCK5_MLP_OUT_INPUT_FEATURES,
        BLOCK5_MLP_OUT_OUTPUT_FEATURES,
        BLOCK5_MLP_OUT_INPUT_ZERO_POINT
    );

    /* Preserve the INT8 T105 boundary. */
    requantize_i32_to_i8(
        accumulator_workspace,
        projection_workspace,
        count,
        BLOCK5_MLP_OUT_MULTIPLIER,
        BLOCK5_MLP_OUT_SHIFT,
        BLOCK5_MLP_OUT_OUTPUT_ZERO_POINT
    );

    /* Op 72 counterpart: T105 -> INT32 T106, scale 2^-16. */
    requantize_i8_to_i32(
        projection_workspace,
        accumulator_workspace,
        count,
        BLOCK5_RESIDUAL2_BRANCH_MULTIPLIER,
        BLOCK5_RESIDUAL2_BRANCH_SHIFT,
        BLOCK5_MLP_OUT_OUTPUT_ZERO_POINT
    );

    /* Op 73: T106 + retained T73 -> T107. */
    residual_add_i32(
        accumulator_workspace,
        residual_input,
        output,
        count
    );
}

void block5_output_boundary_forward(const int32_t *input, int8_t *output)
{
    requantize_i32_to_i8(input, output,
                         BLOCK5_TOKEN_COUNT * BLOCK5_HIDDEN_SIZE,
                         BLOCK5_OUTPUT_REQUANT_MULTIPLIER,
                         BLOCK5_OUTPUT_REQUANT_SHIFT,
                         BLOCK5_OUTPUT_REQUANT_ZERO_POINT);
}

void block5_forward(const int8_t *input, int8_t *output,
                      TransformerBlockWorkspace *workspace)
{
    block5_ln1_forward(input, workspace->activation, &workspace->ln);
    block5_qkv_forward(workspace->activation, workspace->query, workspace->key,
                         workspace->value, workspace->ln.feature_i32);
    block5_attention_forward(workspace->query, workspace->key, workspace->value,
                               workspace->attention_output, workspace->score_workspace,
                               workspace->ln.feature_i32);

    /* Attention scores are dead; retain Part 4's residual in their buffer. */
    block5_out_residual_forward(workspace->attention_output, input,
                                  workspace->score_workspace, workspace->projection_output,
                                  workspace->ln.feature_i32);
    block5_ln2_forward(workspace->score_workspace, workspace->activation, &workspace->ln);
    block5_mlp_gelu_forward(workspace->activation, workspace->mlp_output, &workspace->mlp);

    /* Part 7 consumes the retained residual and replaces it with the final sum. */
    block5_mlp_out_residual_forward(workspace->mlp_output, workspace->score_workspace,
                                      workspace->score_workspace, workspace->projection_output,
                                      workspace->ln.feature_i32);
    block5_output_boundary_forward(workspace->score_workspace, output);
}
