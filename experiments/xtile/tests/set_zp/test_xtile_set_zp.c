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

/* Raw operands let the test independently control the upper 32 bits. */
static inline void xtile_set_zp_raw(uint64_t a, uint64_t b)
{
    __asm__ volatile (
        ".insn r 0x0b, 3, 0, x0, %0, %1"
        :
        : "r" (a), "r" (b)
    );
}

struct TestCase {
    const char *name;
    uint64_t a_operand;
    uint64_t b_operand;
};

int main(void)
{
    /* INT32 operands, active tile 2x2, M=2, N=2, K=8. */
    const uint64_t configuration =
        UINT64_C(3)          |
        (UINT64_C(1) << 2)  |
        (UINT64_C(1) << 5)  |
        (UINT64_C(2) << 8)  |
        (UINT64_C(2) << 18) |
        (UINT64_C(8) << 28);

    const struct TestCase cases[] = {
        { "zero points before configuration",
          UINT64_C(0), UINT64_C(0) },

        { "positive zero points",
          UINT64_C(17), UINT64_C(9) },

        { "negative A zero point",
          UINT64_C(0xffffffffffffffef), UINT64_C(0) },

        { "negative B zero point",
          UINT64_C(0), UINT64_C(0xfffffffffffffff4) },

        { "both zero points negative",
          UINT64_C(0xffffffffffffff80),
          UINT64_C(0xffffffffffffffff) },

        { "INT32 minimum and maximum",
          UINT64_C(0xffffffff80000000),
          UINT64_C(0x000000007fffffff) },

        { "INT32 maximum and minimum",
          UINT64_C(0x000000007fffffff),
          UINT64_C(0xffffffff80000000) },

        { "raw operands with nonzero upper bits",
          UINT64_C(0x12345678ffffffef),
          UINT64_C(0xabcdef0100000005) },

        { "zero-extended negative low-bit patterns",
          UINT64_C(0x00000000ffffffff),
          UINT64_C(0x0000000080000000) },
    };

    const unsigned int case_count =
        (unsigned int)(sizeof(cases) / sizeof(cases[0]));

    printf("xtile.set_zp smoke test: %u cases\n", case_count);

    for (unsigned int i = 0; i < case_count; ++i) {
        /* Case 0 executes before any configuration instruction. */
        if (i == 1) {
            xtile_cfg(configuration);
        }

        xtile_set_zp_raw(cases[i].a_operand, cases[i].b_operand);

        printf("Case %u: %s PASS (accepted)\n", i, cases[i].name);
    }

    printf("PASS: %u zero-point smoke cases completed\n", case_count);
    return 0;
}