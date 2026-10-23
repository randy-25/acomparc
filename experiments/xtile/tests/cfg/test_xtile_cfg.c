#include <stdint.h>
#include <stdio.h>

static inline void xtile_cfg(uint64_t configuration)
{
    __asm__ volatile (
        ".insn r 0x0b, 0, 0, x0, %0, x0"
        :
        : "r" (configuration)
    );
}

int main(void)
{
    const uint64_t configuration =
        (UINT64_C(7)   << 2)  |  /* Active tile rows: 8. */
        (UINT64_C(7)   << 5)  |  /* Active tile columns: 8. */
        (UINT64_C(32)  << 8)  |  /* M. */
        (UINT64_C(128) << 18) |  /* N. */
        (UINT64_C(128) << 28);   /* K. */

    /* A_TYPE, B_TYPE, and reserved bits remain zero. */
    printf("xtile.cfg smoke test: M=%u N=%u K=%u tile=%ux%u\n",
           32u, 128u, 128u, 8u, 8u);

    xtile_cfg(configuration);

    printf("PASS: %u valid xtile.cfg instruction completed\n", 1u);
    return 0;
}