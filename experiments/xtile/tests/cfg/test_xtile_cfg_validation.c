#include <stdint.h>
#include <stdio.h>

extern void xtile_cfg_test_execute(uint64_t configuration);
extern const unsigned char xtile_cfg_test_insn[];

extern void tohost_exit(uintptr_t code)
    __attribute__((noreturn));

static volatile unsigned int trap_expected;
static volatile unsigned int trap_seen;

uintptr_t handle_trap(
    uintptr_t cause,
    uintptr_t epc,
    uintptr_t regs[32]
)
{
    (void)regs;

    const uintptr_t instruction_address =
        (uintptr_t)xtile_cfg_test_insn;

    if (!trap_expected ||
        cause != 2 ||
        epc != instruction_address ||
        trap_seen != 0) {
        printf("FAIL: unexpected trap, cause=%u\n",
               (unsigned int)cause);
        tohost_exit(1);
    }

    trap_seen = 1;

    /* Our custom instruction is exactly four bytes long. */
    return epc + 4;
}

struct TestCase {
    const char *name;
    uint64_t configuration;
    unsigned int expect_trap;
};

int main(void)
{
    const uint64_t base =
        (UINT64_C(7)   << 2)  |
        (UINT64_C(7)   << 5)  |
        (UINT64_C(32)  << 8)  |
        (UINT64_C(128) << 18) |
        (UINT64_C(128) << 28);

    const uint64_t m_mask = UINT64_C(0x3ff) << 8;
    const uint64_t n_mask = UINT64_C(0x3ff) << 18;
    const uint64_t k_mask = UINT64_C(0x3ff) << 28;
    const uint64_t tile_mask = UINT64_C(0xfc);

    const struct TestCase cases[] = {
        { "INT8/INT8 configuration", base, 0 },

        { "rectangular tile 4x8",
          (base & ~tile_mask) |
          (UINT64_C(3) << 2) |
          (UINT64_C(7) << 5), 0 },

        { "INT32/INT8 configuration", base | UINT64_C(1), 0 },
        { "INT8/INT32 configuration", base | UINT64_C(2), 0 },
        { "INT32/INT32 configuration", base | UINT64_C(3), 0 },

        { "dimensions 1x1x1 and tile 1x1",
          (UINT64_C(1) << 8) |
          (UINT64_C(1) << 18) |
          (UINT64_C(1) << 28), 0 },

        { "reject M=0", base & ~m_mask, 1 },
        { "reject N=0", base & ~n_mask, 1 },
        { "reject K=0", base & ~k_mask, 1 },
        { "reject reserved bit 38", base | (UINT64_C(1) << 38), 1 },
        { "reject reserved bit 63", base | (UINT64_C(1) << 63), 1 },
        { "reject zero configuration", UINT64_C(0), 1 },

        { "valid reconfiguration after invalid cases",
          (base & ~m_mask) | (UINT64_C(16) << 8), 0 },
    };

    const unsigned int case_count =
        (unsigned int)(sizeof(cases) / sizeof(cases[0]));

    unsigned int observed_traps = 0;

    for (unsigned int i = 0; i < case_count; ++i) {
        trap_seen = 0;
        trap_expected = cases[i].expect_trap;

        xtile_cfg_test_execute(cases[i].configuration);

        trap_expected = 0;

        if (trap_seen != cases[i].expect_trap) {
            printf("Case %u: %s FAIL, expected trap=%u observed=%u\n",
                   i, cases[i].name,
                   cases[i].expect_trap, trap_seen);
            return 1;
        }

        observed_traps += trap_seen;

        printf("Case %u: %s PASS (%s)\n",
               i, cases[i].name,
               cases[i].expect_trap ? "expected trap" : "accepted");
    }

    printf("PASS: %u cases, %u expected illegal-instruction traps\n",
           case_count, observed_traps);

    return 0;
}