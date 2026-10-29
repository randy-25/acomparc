#ifndef TINYSTORIES_INTEGER_BLOCK_STACK_H
#define TINYSTORIES_INTEGER_BLOCK_STACK_H
#include "block_workspace.h"
#define TINYSTORIES_BLOCK_COUNT 8u
#define TINYSTORIES_ELEMENTS 4096u
typedef void (*TransformerBlockForward)(const int8_t *, int8_t *, TransformerBlockWorkspace *);
extern const TransformerBlockForward tinystories_block_forwards[TINYSTORIES_BLOCK_COUNT];

/* Actual block-0-through-7 C chain. Input/output/scratch each hold 4096 INT8
 * values. They must be mutually disjoint and separate from the workspace.
 * Outputs alternate between scratch and output; block 7 finishes in output.
 * One workspace is reused across all blocks. No allocation is performed.
 */
void tinystories_blocks_forward(const int8_t *input, int8_t *output,
                                int8_t *scratch, TransformerBlockWorkspace *workspace);
#endif
