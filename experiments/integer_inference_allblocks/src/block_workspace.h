#ifndef TINYSTORIES_BLOCK_WORKSPACE_H
#define TINYSTORIES_BLOCK_WORKSPACE_H
#include "layernorm.h"
#include "mlp_gelu.h"

/* Caller owns every allocation. Row arrays: 32 elements; ordinary activation
 * arrays: 4096; MLP arrays: 16384; scores: 16384 INT32; wide: 4096 INT64.
 * ln.feature_i32 may share mlp.product_i32's 16384-element allocation.
 * ln.fine_i64 may share mlp.wide_i64's 4096-element allocation.
 * Other backing buffers must be mutually disjoint. Input/output buffers must
 * also be disjoint from these allocations and from each other.
 */
typedef struct {
    LayerNormI8Workspace ln;
    MlpGeluWorkspace mlp;
    int8_t *activation;
    int8_t *query;
    int8_t *key;
    int8_t *value;
    int8_t *attention_output;
    int8_t *projection_output;
    int8_t *mlp_output;
    int32_t *score_workspace;
} TransformerBlockWorkspace;

/* After a complete block, score_workspace[0..4095] holds the INT32 final sum
 * at scale 2^-16, before the INT8 boundary conversion. It is overwritten by
 * the next block. Other workspace contents are temporary.
 */
#endif
