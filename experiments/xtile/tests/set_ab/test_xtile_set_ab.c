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

static inline void xtile_set_ab(const void *a, const void *bt)
{
    __asm__ volatile (
        ".insn r 0x0b, 1, 0, x0, %0, %1"
        :
        : "r" ((uintptr_t)a),
          "r" ((uintptr_t)bt)
    );
}

static const int8_t a_first[2][8] = {{1}};
static const int8_t bt_first[2][8] = {{2}};
static const int8_t a_second[2][8] = {{3}};
static const int8_t bt_second[2][8] = {{4}};

int main(void)
{
    /* INT8 operands, active tile 2x2, M=2, N=2, K=8. */
    const uint64_t configuration =
        (UINT64_C(1) << 2)  |
        (UINT64_C(1) << 5)  |
        (UINT64_C(2) << 8)  |
        (UINT64_C(2) << 18) |
        (UINT64_C(8) << 28);

    printf("xtile.set_ab smoke test: %u cases\n", 3u);

    /* No xtile.cfg has executed yet. */
    xtile_set_ab(a_first, bt_first);
    printf("Case %u: set_ab before cfg PASS\n", 0u);

    xtile_cfg(configuration);
    xtile_set_ab(a_first, bt_first);
    printf("Case %u: set_ab after cfg PASS\n", 1u);

    xtile_set_ab(a_second, bt_second);
    printf("Case %u: replacement addresses accepted PASS\n", 2u);

    printf("PASS: %u xtile.set_ab smoke cases completed\n", 3u);
    return 0;
}