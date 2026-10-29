#ifndef TINYSTORIES_ALLBLOCKS_BLOCK6_H
#define TINYSTORIES_ALLBLOCKS_BLOCK6_H
#include "../layernorm.h"
#include "../mlp_gelu.h"
#include "../block_workspace.h"

#define BLOCK6_TOKEN_COUNT 32u
#define BLOCK6_HIDDEN_SIZE 128u

/* Part 1: LN1, [32,128] INT8 input at scale 2^-8, zero point 0.
 * Output uses the saved block-6 LN1 INT8 scale and zero point.
 * Row workspace arrays: 32 elements; feature workspace arrays: 4096.
 * Input, output and all workspace buffers must be mutually disjoint.
 * No allocation is performed. Workspace contents are overwritten.
 */
void block6_ln1_forward(const int8_t *input, int8_t *output,
                         LayerNormI8Workspace *workspace);

/* Part 2: Q/K/V from the block's LN1 INT8 output, [32,128].
 * Each output contains 4096 INT8 values at its saved scale/zero point.
 * Accumulator workspace: 4096 INT32 values, reused between Q, K and V.
 * Input, all outputs, accumulator and parameter arrays must be disjoint.
 * Accumulation and requantization remain separate operations.
 */
void block6_qkv_forward(const int8_t *input,
                         int8_t *q_output, int8_t *k_output, int8_t *v_output,
                         int32_t *accumulator);

#define BLOCK6_HEAD_COUNT 16u
#define BLOCK6_HEAD_DIM 8u
#define BLOCK6_SCORE_ELEMENTS (BLOCK6_HEAD_COUNT * BLOCK6_TOKEN_COUNT * BLOCK6_TOKEN_COUNT)

/* Scores: Q/K INT8 [32,128] -> INT32 [16,32,32], zero point 0.
 * No masking or extra score scaling is performed here.
 * All buffers must be valid and disjoint.
 */
void block6_attention_scores_forward(const int8_t *query, const int8_t *key,
                                      int32_t *scores);

/* Part 3: causal software attention, with XREQUANT context conversion.
 * Q/K/V: INT8 [32,128] at their exported scales and zero points.
 * Output: INT8 [32,128] at the saved context output scale and zero point.
 * score_workspace: 16384 INT32 elements, reused through probabilities.
 * context_workspace: 4096 INT32 elements, merged token-major context.
 * All input, output and workspace buffers must be valid and disjoint.
 * Workspace contents are overwritten. No allocation is performed.
 */
void block6_attention_forward(const int8_t *query, const int8_t *key,
                               const int8_t *value, int8_t *output,
                               int32_t *score_workspace, int32_t *context_workspace);

/* Part 4: output projection and the first residual addition.
 * attention_output: INT8 [32,128], saved Part-3 output quantization.
 * residual_input: original block INT8 input, zero point 0.
 * output: INT32 [32,128], scale 2^-16, zero point 0.
 * projection_workspace: 4096 INT8 values, saved projection output.
 * accumulator_workspace: 4096 INT32 values.
 * output may equal accumulator_workspace. All other buffers must be disjoint.
 * Caller provides valid buffers; no allocation is performed.
 */
void block6_out_residual_forward(const int8_t *attention_output,
                                  const int8_t *residual_input, int32_t *output,
                                  int8_t *projection_workspace,
                                  int32_t *accumulator_workspace);

/* Part 5: LN2, INT32 [32,128] input at scale 2^-16, zero point 0.
 * Output: INT8 [32,128] at the saved block-specific LN2 quantization.
 * Row workspace arrays: 32 elements; feature arrays: 4096 elements.
 * Input must remain available until normalization completes. Input, output
 * and all workspace buffers must be valid and mutually disjoint.
 * No allocation is performed. Workspace contents are overwritten.
 */
void block6_ln2_forward(const int32_t *input, int8_t *output,
                         LayerNormI8Workspace *workspace);

/* Part 6: 128-to-512 INT8 expansion, including bias, then staged integer GELU.
 * input: 4096 INT8 LN2 output values; output: 16384 INT8 GELU values.
 * All input, output and workspace buffers must be valid and disjoint.
 * No allocation is performed. The expansion INT8 boundary is preserved.
 */
void block6_mlp_gelu_forward(const int8_t *input, int8_t *output,
                              MlpGeluWorkspace *workspace);

/* Expansion alone. Accumulator/output capacities: 16384 values each. */
void block6_mlp_expand_forward(const int8_t *input, int8_t *output,
                                int32_t *accumulator);

/* GELU alone, 16384 INT8 inputs/outputs. Input may equal workspace->t89;
 * otherwise it must be disjoint from every workspace buffer and the output.
 * Workspace buffers and output must be mutually disjoint.
 */
void block6_gelu_forward(const int8_t *input, int8_t *output,
                          MlpGeluWorkspace *workspace);

/* Part 7: 512-to-128 projection and addition of the retained Part-4 residual.
 * input: 16384 INT8 GELU values. residual_input: 4096 INT32 values at 2^-16.
 * output: 4096 INT32 values at 2^-16. projection_workspace: 4096 INT8 values.
 * accumulator_workspace: 4096 INT32 values.
 * Input, residual_input and both workspaces must be valid and mutually disjoint.
 * Output may equal residual_input or accumulator_workspace; otherwise keep it
 * separate from every supplied buffer. No allocation is performed.
 */
void block6_mlp_out_residual_forward(const int8_t *input,
                                      const int32_t *residual_input, int32_t *output,
                                      int8_t *projection_workspace,
                                      int32_t *accumulator_workspace);

/* Final block boundary: 4096 INT32 values at 2^-16 -> INT8 at the saved
 * output scale, zero point 0. XREQUANT mode 2 retains the profile's rounding
 * and saturation. Input/output buffers must be valid and disjoint.
 */
void block6_output_boundary_forward(const int32_t *input, int8_t *output);

/* Complete block, INT8 [32,128] input and output at the exported boundary
 * scales. Input/output/workspace must satisfy block_workspace.h's rules.
 * No allocation or floating-point activation arithmetic is performed.
 */
void block6_forward(const int8_t *input, int8_t *output,
                      TransformerBlockWorkspace *workspace);
#endif
