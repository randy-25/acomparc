#include "model.h"
#include "blocks/block0.h"
#include "blocks/block1.h"
#include "blocks/block2.h"
#include "blocks/block3.h"
#include "blocks/block4.h"
#include "blocks/block5.h"
#include "blocks/block6.h"
#include "blocks/block7.h"

const TransformerBlockForward tinystories_block_forwards[TINYSTORIES_BLOCK_COUNT] = {
    block0_forward, block1_forward, block2_forward, block3_forward,
    block4_forward, block5_forward, block6_forward, block7_forward
};

void tinystories_blocks_forward(const int8_t *input, int8_t *output,
                                int8_t *scratch, TransformerBlockWorkspace *workspace)
{
    const int8_t *current = input;
    for (size_t block = 0; block < TINYSTORIES_BLOCK_COUNT; ++block) {
        int8_t *next = (block & 1u) ? output : scratch;
        tinystories_block_forwards[block](current, next, workspace);
        current = next;
    }
}
