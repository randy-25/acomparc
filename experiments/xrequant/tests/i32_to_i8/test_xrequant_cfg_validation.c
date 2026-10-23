#include <stdint.h>
#include <stdio.h>

extern void xrequant_cfg_test_execute(uint64_t cfg);
extern const unsigned char xrequant_cfg_test_insn[];

extern void tohost_exit(uintptr_t code)
    __attribute__((noreturn));

static volatile unsigned int trap_expected;
static volatile unsigned int trap_seen;

static unsigned int cfg_cases;
static unsigned int expected_traps;
static unsigned int run_checks;

static const int32_t input[2] = {20, -20};

/* Guards immediately surround the two destination bytes. */
static int8_t output[4];

uintptr_t handle_trap(
    uintptr_t cause,
    uintptr_t epc,
    uintptr_t regs[32]
)
{
    (void)regs;

    uintptr_t tval;
    __asm__ volatile("csrr %0, mtval" : "=r"(tval));

    /*
     * CFG with rs1=a0 has instruction bits 0x0005002b.
     * mtval contains those instruction bits, not the CFG value.
     */
    if (!trap_expected ||
        trap_seen ||
        cause != 2 ||
        epc != (uintptr_t)xrequant_cfg_test_insn ||
        tval != UINT32_C(0x0005002b)) {
        printf(
            "FAIL: unexpected trap: cause=%u epc=%llx tval=%llx\n",
            (unsigned int)cause,
            (unsigned long long)epc,
            (unsigned long long)tval
        );
        tohost_exit(1);
    }

    trap_seen = 1;
    return epc + 4;
}

static uint64_t pack_cfg(
    unsigned int mode,
    unsigned int shift,
    uint32_t count
)
{
    return (uint64_t)mode |
           ((uint64_t)shift << 3) |
           ((uint64_t)count << 9);
}

static int check_cfg(
    const char *name,
    uint64_t cfg,
    unsigned int should_trap
)
{
    trap_seen = 0;
    trap_expected = should_trap;

    xrequant_cfg_test_execute(cfg);

    trap_expected = 0;

    if (trap_seen != should_trap) {
        printf("FAIL CFG case %u: %s\n", cfg_cases, name);
        return 1;
    }

    printf(
        "CFG case %u: %s PASS (%s)\n",
        cfg_cases, name,
        should_trap ? "expected trap" : "accepted"
    );

    ++cfg_cases;
    expected_traps += should_trap;

    return 0;
}

static void setup_operands(void)
{
    const int64_t multiplier = 3;
    const intptr_t output_zp = -5;

    __asm__ volatile(
        ".insn r 0x2b, 1, 0, x0, %0, %1\n\t"
        ".insn r 0x2b, 2, 0, x0, %2, x0\n\t"
        ".insn r 0x2b, 3, 0, x0, x0, %3"
        :
        : "r"((uintptr_t)input),
          "r"((uintptr_t)&output[1]),
          "r"(multiplier),
          "r"(output_zp)
        : "memory"
    );
}

static int check_retained_cfg(void)
{
    output[0] = 0x35;
    output[1] = 0x55;
    output[2] = 0x55;
    output[3] = 0x6a;

    __asm__ volatile(
        ".insn r 0x2b, 4, 0, x0, x0, x0"
        : : : "memory"
    );

    /*
     * Requires the retained configuration to remain:
     * MODE=2, SHIFT=3, COUNT=2.
     */
    if (output[0] != 0x35 ||
        output[1] != 3 ||
        output[2] != -12 ||
        output[3] != 0x6a) {
        printf(
            "FAIL: retained CFG/output check: [%d,%d,%d,%d]\n",
            (int)output[0], (int)output[1],
            (int)output[2], (int)output[3]
        );
        return 1;
    }

    ++run_checks;
    return 0;
}

#define CHECK_CFG(name, cfg, trap)             \
    do {                                      \
        if (check_cfg(name, cfg, trap)) {      \
            return 1;                         \
        }                                     \
    } while (0)

int main(void)
{
    /* Acceptance checks only: do not RUN the maximum count. */
    CHECK_CFG("minimum count and shift", pack_cfg(2, 1, 1), 0);
    CHECK_CFG("maximum shift", pack_cfg(2, 63, 1), 0);
    CHECK_CFG(
        "maximum encoded count",
        pack_cfg(2, 39, UINT32_MAX), 0
    );

    CHECK_CFG("mode 0 accepted", pack_cfg(0, 3, 2), 0);
    CHECK_CFG("mode 1 accepted", pack_cfg(1, 3, 2), 0);
    CHECK_CFG("mode 3 accepted", pack_cfg(3, 3, 2), 0);
    const uint64_t valid_cfg = pack_cfg(2, 3, 2);
    CHECK_CFG("mode 4 accepted", pack_cfg(4, 6, 2), 0);
    CHECK_CFG("mode 4 second shift zero", pack_cfg(4, 0, 2), 0);
    CHECK_CFG("mode 4 second shift 31", pack_cfg(4, 31, 2), 0);
    CHECK_CFG("ordinary configuration", valid_cfg, 0);

    setup_operands();

    if (check_retained_cfg()) {
        return 1;
    }

    for (unsigned int mode = 0; mode < 8; ++mode) {
        if (mode <= 4) {
            continue;
        }

        printf("Checking rejected mode %u\n", mode);

        CHECK_CFG("unsupported mode", pack_cfg(mode, 3, 2), 1);

        if (check_retained_cfg()) {
            return 1;
        }
    }
    CHECK_CFG("mode 4 rejects second shift 32", pack_cfg(4, 32, 2), 1);
    if (check_retained_cfg()) {
        return 1;
    }

    CHECK_CFG("mode 4 rejects second shift 63", pack_cfg(4, 63, 2), 1);
    if (check_retained_cfg()) {
        return 1;
    }

    CHECK_CFG("zero shift", pack_cfg(2, 0, 2), 1);
    if (check_retained_cfg()) {
        return 1;
    }

    CHECK_CFG("zero count", pack_cfg(2, 3, 0), 1);
    if (check_retained_cfg()) {
        return 1;
    }

    CHECK_CFG(
        "reserved bit 41",
        valid_cfg | (UINT64_C(1) << 41), 1
    );
    if (check_retained_cfg()) {
        return 1;
    }

    CHECK_CFG(
        "reserved bit 63",
        valid_cfg | (UINT64_C(1) << 63), 1
    );
    if (check_retained_cfg()) {
        return 1;
    }

    CHECK_CFG("valid CFG after rejections", valid_cfg, 0);
    if (check_retained_cfg()) {
        return 1;
    }

    printf(
        "PASS: %u CFG cases, %u expected traps; "
        "%u RUN checks passed\n",
        cfg_cases, expected_traps, run_checks
    );

    return 0;
}

#undef CHECK_CFG