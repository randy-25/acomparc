#ifndef INTEGER_INFERENCE_BLOCK0_H
#define INTEGER_INFERENCE_BLOCK0_H

#include "../layernorm.h"

#define BLOCK0_TOKEN_COUNT 32u
#define BLOCK0_HIDDEN_SIZE 128u


#define BLOCK0_HEAD_COUNT 16u
#define BLOCK0_HEAD_DIM 8u
#define BLOCK0_SCORE_ELEMENTS (BLOCK0_HEAD_COUNT * BLOCK0_TOKEN_COUNT * BLOCK0_TOKEN_COUNT)



/*
 * Block-0 first LayerNorm, Op 0–14.
 *
 * Input:  INT8 [1,32,128], scale 2^-8, zero point 0.
 * Output: INT8 [1,32,128], scale 0.03362198546528816,
 *         zero point -17.
 *
 * Workspace capacities:
 * - mean_sum, variance, inv_std: 32 elements each.
 * - feature_i32, fine_i64: 4096 elements each.
 *
 * All input/output/workspace buffers must be valid and
 * mutually non-overlapping.
 * No memory allocation is performed.
 */
void block0_ln1_forward(
    const int8_t *input,
    int8_t *output,
    LayerNormI8Workspace *workspace
);

/*
 * Block-0 Q/K/V projections, Op 15–17.
 *
 * input: T45, INT8 [1,32,128], zero point -17.
 * q_output: T46, INT8 [1,32,128], zero point -12.
 * k_output: T47, INT8 [1,32,128], zero point -30.
 * v_output: T48, INT8 [1,32,128], zero point 4.
 *
 * Each input/output buffer contains 4096 elements.
 * accumulator contains 4096 INT32 elements.
 * All buffers must be valid and mutually non-overlapping.
 * No memory allocation is performed.
 */
void block0_qkv_forward(
    const int8_t *input,
    int8_t *q_output,
    int8_t *k_output,
    int8_t *v_output,
    int32_t *accumulator
);

/*
 * Q/K inputs: INT8 [32,128], original token-major layout.
 * Output: INT32 [16,32,32], indexed [head, query, key].
 * Score scale: 0.0004575034413775367; zero point: 0.
 * Buffers must be valid and non-overlapping.
 */
void block0_attention_scores_forward(
    const int8_t *query,
    const int8_t *key,
    int32_t *scores
);

/*
 * Block-0 attention, from projected Q/K/V through T70.
 *
 * query/key/value: INT8 [32,128], projection outputs.
 * output:          INT8 [32,128], T70.
 *
 * score_workspace:   INT32 [16,32,32], 16384 elements.
 * context_workspace: INT32 [32,128],    4096 elements.
 *
 * All supplied buffers must be valid and non-overlapping.
 * Both workspaces are overwritten.
 */
void block0_attention_forward(
    const int8_t *query,
    const int8_t *key,
    const int8_t *value,
    int8_t *output,
    int32_t *score_workspace,
    int32_t *context_workspace
);

/*
 * Block-0 output projection and first residual addition.
 *
 * attention_output: INT8 [32,128], T70.
 * residual_input:   INT8 [32,128], original T0.
 * output:           INT32 [32,128], T73, scale 2^-16.
 *
 * projection_workspace: INT8 [4096], holds T71.
 * accumulator_workspace: INT32 [4096], holds projection accumulators.
 *
 * output may equal accumulator_workspace.
 * All other supplied buffers must not overlap.
 */
void block0_out_residual_forward(
    const int8_t *attention_output,
    const int8_t *residual_input,
    int32_t *output,
    int8_t *projection_workspace,
    int32_t *accumulator_workspace
);

/*
 * Block-0 LN2, Op 43-57.
 *
 * Input:  INT32 [32,128], T73, scale 2^-16, zero point 0.
 * Output: INT8  [32,128], T88.
 *
 * Workspace:
 *   mean_sum, variance, inv_std: 32 INT32 elements each.
 *   feature_i32: 4096 INT32 elements.
 *   fine_i64:    4096 INT64 elements.
 *
 * Input, output, and workspace buffers must not overlap.
 * Workspace buffers must also be mutually non-overlapping.
 * The original input remains needed through normalization.
 */
void block0_ln2_forward(
    const int32_t *input,
    int8_t *output,
    LayerNormI8Workspace *workspace
);

#define BLOCK0_GELU_ADD_CHUNK_ELEMENTS 4096u

typedef struct {
    int8_t *t89;          /* 16384 elements: expansion output */
    int8_t *half;         /* 16384 elements: T92 */
    int8_t *branch;       /* 16384 elements: T94 -> T96 -> T97 -> T99 -> T103 */
    int32_t *product_i32; /* 16384 elements: accumulator/products/T101/T102 */
    int64_t *wide_i64;    /* BLOCK0_GELU_ADD_CHUNK_ELEMENTS elements */
} Block0MlpGeluWorkspace;


/*
 * Block-0 Part 6, Op 58-70.
 *
 * Input:  INT8 [32,128], T88.
 * Output: INT8 [32,512], T104.
 *
 * All input, output, and workspace buffers must be valid
 * and mutually non-overlapping.
 *
 * Workspace contents are overwritten.
 */
void block0_mlp_gelu_forward(
    const int8_t *input,
    int8_t *output,
    Block0MlpGeluWorkspace *workspace
);

void block0_mlp_expand_forward(
    const int8_t *input,
    int8_t *output,
    int32_t *accumulator
);

/*
 * Block-0 Part 7, Op 71-73.
 *
 * input:          INT8  [32,512], T104.
 * residual_input: INT32 [32,128], T73, scale 2^-16.
 * output:         INT32 [32,128], T107, scale 2^-16.
 *
 * projection_workspace: 4096 INT8 elements, T105.
 * accumulator_workspace: 4096 INT32 elements, accumulator then T106.
 *
 * Input, residual, and workspace buffers must not overlap.
 * Output may equal residual_input or accumulator_workspace;
 * otherwise, output must not overlap any supplied buffer.
 */
void block0_mlp_out_residual_forward(
    const int8_t *input,
    const int32_t *residual_input,
    int32_t *output,
    int8_t *projection_workspace,
    int32_t *accumulator_workspace
);

/*
 * Caller-provided buffers for the complete block.
 *
 * ln:
 *   mean_sum, variance, inv_std: 32 INT32 elements each.
 *   feature_i32: at least 4096 INT32 elements.
 *   fine_i64: at least 4096 INT64 elements.
 *
 * mlp:
 *   t89, half, branch: 16384 INT8 elements each.
 *   product_i32: 16384 INT32 elements.
 *   wide_i64: BLOCK0_GELU_ADD_CHUNK_ELEMENTS INT64 elements.
 *
 * Permitted backing-buffer reuse:
 *   ln.feature_i32 and mlp.product_i32 may share the larger allocation.
 *   ln.fine_i64 and mlp.wide_i64 may share an allocation large
 *   enough for both requirements.
 *
 * Other backing buffers must be mutually non-overlapping.
 */
typedef struct {
    LayerNormI8Workspace ln;
    Block0MlpGeluWorkspace mlp;

    int8_t *activation;        /* 4096: T45, then T88 */
    int8_t *query;             /* 4096 */
    int8_t *key;               /* 4096 */
    int8_t *value;             /* 4096 */
    int8_t *attention_output;  /* 4096: T70 */
    int8_t *projection_output; /* 4096: T71, then T105 */
    int8_t *mlp_output;        /* 16384: T104 */

    int32_t *score_workspace;  /* 16384: scores, then first 4096 hold T73 */
} Block0Workspace;

/*
 * Complete integer block-0 forward pass.
 *
 * Input:  T0,   INT8  [32,128], scale 2^-8,  zero point 0.
 * Output: T107, INT32 [32,128], scale 2^-16, zero point 0.
 *
 * Input must not overlap output or any workspace buffer.
 *
 * Output may equal workspace->score_workspace or
 * workspace->ln.feature_i32. Otherwise, keep output
 * separate from all workspace buffers.
 *
 * Workspace contents are overwritten.
 * No tensor allocation occurs inside this function.
 */
void block0_forward(
    const int8_t *input,
    int32_t *output,
    Block0Workspace *workspace
);

#endif