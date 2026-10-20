#include "block0.h"
#include "../requantization.h"
#include "../../params/block0_params.h"
#include "../projection.h"
#include "../attention.h"
#include "../residual.h"
#include "../gelu.h"

void block0_ln1_forward(
    const int8_t *input,
    int8_t *output,
    LayerNormI8Workspace *workspace
)
{
    const size_t token_count = BLOCK0_TOKEN_COUNT;
    const size_t feature_count = BLOCK0_HIDDEN_SIZE;
    const size_t element_count = token_count * feature_count;

    const LayerNormI8Params *params = &block0_ln1_params;

    /* Op 0–1: mean sums; broadcast through indexing. */
    layernorm_sum_i8(
        input, workspace->mean_sum,
        token_count, feature_count
    );

    /* Op 2: fine squared differences, scale 2^-30. */
    layernorm_squared_difference_i8(
        input, workspace->mean_sum, workspace->fine_i64,
        token_count, feature_count
    );

    /* T33: INT64 at 2^-30 -> INT32 at 2^-15. */
    requantize_shift_i64_to_i32(
        workspace->fine_i64, workspace->feature_i32,
        element_count, params->square_shift
    );

    /* Op 3: variance, scale 2^-22. */
    layernorm_variance_i32(
        workspace->feature_i32, workspace->variance,
        token_count, feature_count
    );

    /* Op 4: variance + epsilon; reuse variance buffer. */
    layernorm_add_epsilon_i32(
        workspace->variance, workspace->variance,
        token_count, params->epsilon_q
    );

    /* Op 5: inverse standard deviation, scale 2^-16. */
    layernorm_rsqrt_i32(
        workspace->variance, workspace->inv_std,
        token_count, params->rsqrt_numerator
    );

    /* Op 6–11: fine normalization, scale 2^-31. */
    layernorm_normalize_i8(
        input, workspace->mean_sum, workspace->inv_std,
        workspace->fine_i64, token_count, feature_count
    );

    /* T42: INT64 at 2^-31 -> INT32 at 2^-16. */
    requantize_shift_i64_to_i32(
        workspace->fine_i64, workspace->feature_i32,
        element_count, params->normalize_shift
    );

    /* Op 12–13: affine transform, scale 2^-32. */
    layernorm_affine_i32(
        workspace->feature_i32, params->gamma, params->beta,
        workspace->fine_i64, token_count, feature_count,
        params->beta_alignment
    );

    /* Op 14: T44 -> INT8 T45. */
    requantize_i64_to_i8(
        workspace->fine_i64, output, element_count,
        params->output_multiplier, params->output_shift,
        params->output_zero_point
    );
}


void block0_qkv_forward(
    const int8_t *input,
    int8_t *q_output,
    int8_t *k_output,
    int8_t *v_output,
    int32_t *accumulator
)
{
    const size_t tokens = BLOCK0_TOKEN_COUNT;
    const size_t features = BLOCK0_HIDDEN_SIZE;
    const size_t elements = tokens * features;

    /* Op 15: Q accumulation, then requantization to T46. */
    projection_accumulate_i8(
        input, block0_q_weights, NULL, accumulator,
        tokens, features, features,
        BLOCK0_QKV_INPUT_ZERO_POINT
    );

    requantize_i32_to_i8(
        accumulator, q_output, elements,
        BLOCK0_Q_MULTIPLIER,
        BLOCK0_QKV_REQUANT_SHIFT,
        BLOCK0_Q_OUTPUT_ZERO_POINT
    );

    /* Op 16: reuse accumulator for K. */
    projection_accumulate_i8(
        input, block0_k_weights, NULL, accumulator,
        tokens, features, features,
        BLOCK0_QKV_INPUT_ZERO_POINT
    );

    requantize_i32_to_i8(
        accumulator, k_output, elements,
        BLOCK0_K_MULTIPLIER,
        BLOCK0_QKV_REQUANT_SHIFT,
        BLOCK0_K_OUTPUT_ZERO_POINT
    );

    /* Op 17: reuse accumulator for V. */
    projection_accumulate_i8(
        input, block0_v_weights, NULL, accumulator,
        tokens, features, features,
        BLOCK0_QKV_INPUT_ZERO_POINT
    );

    requantize_i32_to_i8(
        accumulator, v_output, elements,
        BLOCK0_V_MULTIPLIER,
        BLOCK0_QKV_REQUANT_SHIFT,
        BLOCK0_V_OUTPUT_ZERO_POINT
    );
}

void block0_attention_scores_forward(
    const int8_t *query,
    const int8_t *key,
    int32_t *scores
)
{
    attention_scores_i8(
        query,
        key,
        scores,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_HEAD_COUNT,
        BLOCK0_HEAD_DIM,
        BLOCK0_Q_OUTPUT_ZERO_POINT,
        BLOCK0_K_OUTPUT_ZERO_POINT
    );
}

void block0_attention_forward(
    const int8_t *query,
    const int8_t *key,
    const int8_t *value,
    int8_t *output,
    int32_t *score_workspace,
    int32_t *context_workspace
)
{
    /* Q × K^T, using the original projection layouts. */
    block0_attention_scores_forward(
        query,
        key,
        score_workspace
    );

    /* Reuse the score buffer for shifted scores. */
    attention_causal_shift_i32(
        score_workspace,
        score_workspace,
        BLOCK0_HEAD_COUNT,
        BLOCK0_TOKEN_COUNT
    );

    /* Reuse it for exponentials. */
    attention_exp_lut_i32(
        score_workspace,
        score_workspace,
        BLOCK0_HEAD_COUNT,
        BLOCK0_TOKEN_COUNT,
        block0_softmax_exp_lut,
        BLOCK0_SOFTMAX_COORD_MULTIPLIER,
        BLOCK0_SOFTMAX_COORD_SHIFT
    );

    /* Reuse it for probabilities. */
    attention_normalize_exp_i32(
        score_workspace,
        score_workspace,
        BLOCK0_HEAD_COUNT,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_SOFTMAX_PROB_FRAC_BITS
    );

    /* Probability × V; write directly in token-major order. */
    attention_context_i8(
        score_workspace,
        value,
        context_workspace,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_HEAD_COUNT,
        BLOCK0_HEAD_DIM,
        BLOCK0_V_OUTPUT_ZERO_POINT
    );

    /* Separate requantization produces the final T70 layout. */
    requantize_i32_to_i8(
        context_workspace,
        output,
        BLOCK0_TOKEN_COUNT * BLOCK0_HIDDEN_SIZE,
        BLOCK0_ATTN_CONTEXT_MULTIPLIER,
        BLOCK0_ATTN_CONTEXT_SHIFT,
        BLOCK0_ATTN_CONTEXT_OUTPUT_ZERO_POINT
    );
}

void block0_out_residual_forward(
    const int8_t *attention_output,
    const int8_t *residual_input,
    int32_t *output,
    int8_t *projection_workspace,
    int32_t *accumulator_workspace
)
{
    const size_t count = BLOCK0_TOKEN_COUNT * BLOCK0_HIDDEN_SIZE;

    projection_accumulate_i8(
        attention_output,
        block0_out_weights,
        block0_out_bias,
        accumulator_workspace,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_HIDDEN_SIZE,
        BLOCK0_HIDDEN_SIZE,
        BLOCK0_OUT_PROJ_INPUT_ZERO_POINT
    );

    /* Preserve the notebook's INT8 T71 boundary. */
    requantize_i32_to_i8(
        accumulator_workspace,
        projection_workspace,
        count,
        BLOCK0_OUT_PROJ_MULTIPLIER,
        BLOCK0_OUT_PROJ_SHIFT,
        BLOCK0_OUT_PROJ_OUTPUT_ZERO_POINT
    );

    /* Op 41 counterpart: T71 -> INT32 T72 at scale 2^-16. */
    requantize_i8_to_i32(
        projection_workspace,
        output,
        count,
        BLOCK0_RESIDUAL1_BRANCH_MULTIPLIER,
        BLOCK0_RESIDUAL1_BRANCH_SHIFT,
        BLOCK0_OUT_PROJ_OUTPUT_ZERO_POINT
    );

    /* Op 42: exact T0 alignment, followed by addition. */
    residual_add_i8_i32(
        residual_input,
        output,
        output,
        count,
        BLOCK0_RESIDUAL1_INPUT_ALIGNMENT,
        BLOCK0_RESIDUAL1_INPUT_ZERO_POINT
    );
}

void block0_ln2_forward(
    const int32_t *input,
    int8_t *output,
    LayerNormI8Workspace *workspace
)
{
    const LayerNormI8Params *params = &block0_ln2_params;

    const size_t token_count = BLOCK0_TOKEN_COUNT;
    const size_t feature_count = BLOCK0_HIDDEN_SIZE;
    const size_t count = token_count * feature_count;

    /*
     * Op 43-44: mean sum and broadcast indexing.
     * T74/T75: INT32, scale 2^-23.
     */
    layernorm_sum_i32(
        input,
        workspace->mean_sum,
        token_count,
        feature_count
    );

    /*
     * Op 45: fine squared differences.
     * INT64, scale 2^-46.
     */
    layernorm_squared_difference_i32(
        input,
        workspace->mean_sum,
        workspace->fine_i64,
        token_count,
        feature_count
    );

    /* T76: convert to INT32, scale 2^-23. */
    requantize_shift_i64_to_i32(
        workspace->fine_i64,
        workspace->feature_i32,
        count,
        params->square_shift
    );

    /*
     * Op 46: store the feature sum as variance.
     * T77: INT32, scale 2^-30.
     */
    layernorm_variance_i32(
        workspace->feature_i32,
        workspace->variance,
        token_count,
        feature_count
    );

    /* Op 47: T78, variance plus epsilon, scale 2^-30. */
    layernorm_add_epsilon_i32(
        workspace->variance,
        workspace->variance,
        token_count,
        params->epsilon_q
    );

    /* Op 48: T79, inverse standard deviation, scale 2^-16. */
    layernorm_rsqrt_i32(
        workspace->variance,
        workspace->inv_std,
        token_count,
        params->rsqrt_numerator
    );

    /*
     * Op 49-54: normalized features before conversion.
     * Reuse fine_i64; its squared differences are no longer needed.
     * Fine T85: INT64, scale 2^-39.
     */
    layernorm_normalize_i32(
        input,
        workspace->mean_sum,
        workspace->inv_std,
        workspace->fine_i64,
        token_count,
        feature_count
    );

    /*
     * T85: convert to INT32, scale 2^-16.
     * Reuse feature_i32; squared differences are no longer needed.
     */
    requantize_shift_i64_to_i32(
        workspace->fine_i64,
        workspace->feature_i32,
        count,
        params->normalize_shift
    );

    /*
     * Op 55-56: gamma multiplication and aligned beta addition.
     * T87: INT64, scale 2^-32.
     */
    layernorm_affine_i32(
        workspace->feature_i32,
        params->gamma,
        params->beta,
        workspace->fine_i64,
        token_count,
        feature_count,
        params->beta_alignment
    );

    /* Op 57: T88, INT8 MLP input. */
    requantize_i64_to_i8(
        workspace->fine_i64,
        output,
        count,
        params->output_multiplier,
        params->output_shift,
        params->output_zero_point
    );
}

void block0_mlp_expand_forward(
    const int8_t *input,
    int8_t *output,
    int32_t *accumulator
)
{
    projection_accumulate_i8(
        input,
        block0_mlp_expand_weights,
        block0_mlp_expand_bias,
        accumulator,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_MLP_EXPAND_INPUT_FEATURES,
        BLOCK0_MLP_EXPAND_OUTPUT_FEATURES,
        BLOCK0_MLP_EXPAND_INPUT_ZERO_POINT
    );

    requantize_i32_to_i8(
        accumulator,
        output,
        BLOCK0_TOKEN_COUNT * BLOCK0_MLP_EXPAND_OUTPUT_FEATURES,
        BLOCK0_MLP_EXPAND_MULTIPLIER,
        BLOCK0_MLP_EXPAND_SHIFT,
        BLOCK0_MLP_EXPAND_OUTPUT_ZERO_POINT
    );
}

void block0_mlp_gelu_forward(
    const int8_t *input,
    int8_t *output,
    Block0MlpGeluWorkspace *workspace
)
{
    const size_t count =
        BLOCK0_TOKEN_COUNT * BLOCK0_MLP_EXPAND_OUTPUT_FEATURES;

    /* Op 58: expansion, including its INT8 T89 boundary. */
    block0_mlp_expand_forward(
        input,
        workspace->t89,
        workspace->product_i32
    );

    /* Op 59-60: half-input branch -> T92. */
    gelu_multiply_scalar_i8(
        workspace->t89, workspace->product_i32, count,
        BLOCK0_GELU_HALF_INPUT_ZERO_POINT,
        BLOCK0_GELU_HALF_CENTERED_CONSTANT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->half, count,
        BLOCK0_GELU_HALF_MULTIPLIER,
        BLOCK0_GELU_HALF_SHIFT,
        BLOCK0_GELU_HALF_OUTPUT_ZERO_POINT
    );

    /* Op 61-62: signed cube -> T94. */
    gelu_cube_i8(
        workspace->t89, workspace->product_i32, count,
        BLOCK0_GELU_CUBE_INPUT_ZERO_POINT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK0_GELU_CUBE_MULTIPLIER,
        BLOCK0_GELU_CUBE_SHIFT,
        BLOCK0_GELU_CUBE_OUTPUT_ZERO_POINT
    );

    /* Op 63: weighted cubic term -> T96. */
    gelu_multiply_scalar_i8(
        workspace->branch, workspace->product_i32, count,
        BLOCK0_GELU_WEIGHTED_CUBE_INPUT_ZERO_POINT,
        BLOCK0_GELU_WEIGHTED_CUBE_CENTERED_CONSTANT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK0_GELU_WEIGHTED_CUBE_MULTIPLIER,
        BLOCK0_GELU_WEIGHTED_CUBE_SHIFT,
        BLOCK0_GELU_WEIGHTED_CUBE_OUTPUT_ZERO_POINT
    );

    /* Op 64: combine branches, rounding once, in chunks -> T97. */
    for (size_t offset = 0; offset < count;
         offset += BLOCK0_GELU_ADD_CHUNK_ELEMENTS) {
        size_t chunk = count - offset;

        if (chunk > BLOCK0_GELU_ADD_CHUNK_ELEMENTS) {
            chunk = BLOCK0_GELU_ADD_CHUNK_ELEMENTS;
        }

        gelu_add_scaled_i8(
            workspace->t89 + offset,
            workspace->branch + offset,
            workspace->wide_i64,
            chunk,
            BLOCK0_GELU_ADD_LINEAR_ZERO_POINT,
            BLOCK0_GELU_ADD_CUBIC_ZERO_POINT,
            BLOCK0_GELU_ADD_LINEAR_MULTIPLIER,
            BLOCK0_GELU_ADD_CUBIC_MULTIPLIER
        );

        requantize_i64_to_i8(
            workspace->wide_i64,
            workspace->branch + offset,
            chunk,
            INT64_C(1),
            BLOCK0_GELU_ADD_SHIFT,
            BLOCK0_GELU_ADD_OUTPUT_ZERO_POINT
        );
    }

    /* Op 65: quantized tanh argument -> T99. */
    gelu_multiply_scalar_i8(
        workspace->branch, workspace->product_i32, count,
        BLOCK0_GELU_ARGUMENT_INPUT_ZERO_POINT,
        BLOCK0_GELU_ARGUMENT_CENTERED_CONSTANT
    );

    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK0_GELU_ARGUMENT_MULTIPLIER,
        BLOCK0_GELU_ARGUMENT_SHIFT,
        BLOCK0_GELU_ARGUMENT_OUTPUT_ZERO_POINT
    );

    /* Op 66-67: tanh lookup -> T101. */
    gelu_tanh_lut_i8(
        workspace->branch,
        workspace->product_i32,
        count,
        block0_gelu_tanh_lut
    );

    /* Op 68: add integer one -> T102. */
    gelu_add_scalar_i32(
        workspace->product_i32,
        workspace->product_i32,
        count,
        BLOCK0_GELU_TANH_ONE
    );

    /* Op 69: quantized one-plus-tanh factor -> T103. */
    requantize_i32_to_i8(
        workspace->product_i32, workspace->branch, count,
        BLOCK0_GELU_FACTOR_MULTIPLIER,
        BLOCK0_GELU_FACTOR_SHIFT,
        BLOCK0_GELU_FACTOR_OUTPUT_ZERO_POINT
    );

    /* Op 70: centered T92 * centered T103. */
    gelu_multiply_i8(
        workspace->half,
        workspace->branch,
        workspace->product_i32,
        count,
        BLOCK0_GELU_OUTPUT_HALF_ZERO_POINT,
        BLOCK0_GELU_OUTPUT_FACTOR_ZERO_POINT
    );

    /* Selected two-stage rounding -> T104. */
    requantize_two_round_i32_to_i8(
        workspace->product_i32,
        output,
        count,
        BLOCK0_GELU_OUTPUT_Q31_MULTIPLIER,
        BLOCK0_GELU_OUTPUT_SECOND_SHIFT,
        BLOCK0_GELU_OUTPUT_ZERO_POINT
    );
}

void block0_mlp_out_residual_forward(
    const int8_t *input,
    const int32_t *residual_input,
    int32_t *output,
    int8_t *projection_workspace,
    int32_t *accumulator_workspace
)
{
    const size_t count =
        BLOCK0_TOKEN_COUNT * BLOCK0_MLP_OUT_OUTPUT_FEATURES;

    /* Op 71: 512 -> 128 projection, including bias. */
    projection_accumulate_i8(
        input,
        block0_mlp_out_weights,
        block0_mlp_out_bias,
        accumulator_workspace,
        BLOCK0_TOKEN_COUNT,
        BLOCK0_MLP_OUT_INPUT_FEATURES,
        BLOCK0_MLP_OUT_OUTPUT_FEATURES,
        BLOCK0_MLP_OUT_INPUT_ZERO_POINT
    );

    /* Preserve the INT8 T105 boundary. */
    requantize_i32_to_i8(
        accumulator_workspace,
        projection_workspace,
        count,
        BLOCK0_MLP_OUT_MULTIPLIER,
        BLOCK0_MLP_OUT_SHIFT,
        BLOCK0_MLP_OUT_OUTPUT_ZERO_POINT
    );

    /* Op 72 counterpart: T105 -> INT32 T106, scale 2^-16. */
    requantize_i8_to_i32(
        projection_workspace,
        accumulator_workspace,
        count,
        BLOCK0_RESIDUAL2_BRANCH_MULTIPLIER,
        BLOCK0_RESIDUAL2_BRANCH_SHIFT,
        BLOCK0_MLP_OUT_OUTPUT_ZERO_POINT
    );

    /* Op 73: T106 + retained T73 -> T107. */
    residual_add_i32(
        accumulator_workspace,
        residual_input,
        output,
        count
    );
}

void block0_forward(
    const int8_t *input,
    int32_t *output,
    Block0Workspace *workspace
)
{
    /* Part 1: T0 -> LN1 -> T45. */
    block0_ln1_forward(
        input,
        workspace->activation,
        &workspace->ln
    );

    /* Part 2: T45 -> Q/K/V. */
    block0_qkv_forward(
        workspace->activation,
        workspace->query,
        workspace->key,
        workspace->value,
        workspace->ln.feature_i32
    );

    /* Part 3: Q/K/V -> attention -> T70. */
    block0_attention_forward(
        workspace->query,
        workspace->key,
        workspace->value,
        workspace->attention_output,
        workspace->score_workspace,
        workspace->ln.feature_i32
    );

    /*
     * Part 4: T70 -> output projection + original T0 -> T73.
     * Attention scores are dead; retain T73 in their buffer.
     */
    block0_out_residual_forward(
        workspace->attention_output,
        input,
        workspace->score_workspace,
        workspace->projection_output,
        workspace->ln.feature_i32
    );

    /*
     * Part 5: T73 -> LN2 -> T88.
     * Reuse the activation buffer; T45 is no longer needed.
     */
    block0_ln2_forward(
        workspace->score_workspace,
        workspace->activation,
        &workspace->ln
    );

    /* Part 6: T88 -> MLP expansion + GELU -> T104. */
    block0_mlp_gelu_forward(
        workspace->activation,
        workspace->mlp_output,
        &workspace->mlp
    );

    /* Part 7: T104 -> final projection + retained T73 -> T107. */
    block0_mlp_out_residual_forward(
        workspace->mlp_output,
        workspace->score_workspace,
        output,
        workspace->projection_output,
        workspace->ln.feature_i32
    );
}