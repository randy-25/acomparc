#include <stdint.h>
#include <stdio.h>

enum {
    ILLEGAL_INSTRUCTION = 2,
    XREQUANT_OVERFLOW = 25,
    UNWRITTEN = 0x55,
    GUARD_COUNT = 8
};

extern void xrequant_run_test_execute(void);
extern const unsigned char xrequant_run_test_insn[];

extern void tohost_exit(uintptr_t code)
    __attribute__((noreturn));

static volatile unsigned int trap_expected;
static volatile unsigned int trap_seen;
static volatile uintptr_t expected_cause;

static unsigned int passed_cases;
static unsigned int expected_traps;

static int32_t input[2] = {2, 2};

static struct {
    int8_t before[GUARD_COUNT];
    int8_t values[2];
    int8_t after[GUARD_COUNT];
} output;

/* Override the common runtime's weak handler. */
uintptr_t handle_trap(
    uintptr_t cause,
    uintptr_t epc,
    uintptr_t regs[32]
)
{
    (void)regs;

    uintptr_t tval;
    __asm__ volatile("csrr %0, mtval" : "=r"(tval));

    if (!trap_expected ||
        trap_seen ||
        cause != expected_cause ||
        epc != (uintptr_t)xrequant_run_test_insn ||
        tval != UINT32_C(0x0000402b)) {

        printf(
            "FAIL: unexpected trap: cause=%u epc=%llx tval=%llx\n",
            (unsigned int)cause,
            (unsigned long long)epc,
            (unsigned long long)tval
        );

        tohost_exit(1);
    }

    trap_seen = 1;

    /* Skip this known 32-bit RUN instruction for testing only. */
    return epc + 4;
}

static inline void set_cfg(unsigned int count, unsigned int shift)
{
    const uint64_t cfg =
        UINT64_C(2) |
        ((uint64_t)shift << 3) |
        ((uint64_t)count << 9);

    __asm__ volatile(
        ".insn r 0x2b, 0, 0, x0, %0, x0"
        : : "r"(cfg)
    );
}

static inline void set_io(void)
{
    __asm__ volatile(
        ".insn r 0x2b, 1, 0, x0, %0, %1"
        : : "r"((uintptr_t)input),
            "r"((uintptr_t)output.values)
    );
}

static inline void set_mul(int64_t multiplier)
{
    __asm__ volatile(
        ".insn r 0x2b, 2, 0, x0, %0, x0"
        : : "r"(multiplier)
    );
}

static inline void set_zp(int32_t input_zp, int32_t output_zp)
{
    __asm__ volatile(
        ".insn r 0x2b, 3, 0, x0, %0, %1"
        : : "r"((intptr_t)input_zp),
            "r"((intptr_t)output_zp)
    );
}

static int check_run(
    const char *name,
    uintptr_t cause,
    int wanted0,
    int wanted1
)
{
    for (unsigned int i = 0; i < GUARD_COUNT; ++i) {
        output.before[i] = 0x35;
        output.after[i] = 0x6a;
    }

    output.values[0] = UNWRITTEN;
    output.values[1] = UNWRITTEN;

    trap_seen = 0;
    expected_cause = cause;
    trap_expected = cause != 0;

    __asm__ volatile("" : : : "memory");
    xrequant_run_test_execute();
    __asm__ volatile("" : : : "memory");

    trap_expected = 0;

    if (cause != 0 && !trap_seen) {
        printf(
            "FAIL case %u: %s did not trap\n",
            passed_cases, name
        );
        return 1;
    }

    if (output.values[0] != wanted0 ||
        output.values[1] != wanted1) {
        printf(
            "FAIL case %u: %s output=[%d,%d] expected=[%d,%d]\n",
            passed_cases, name,
            (int)output.values[0], (int)output.values[1],
            wanted0, wanted1
        );
        return 1;
    }

    for (unsigned int i = 0; i < GUARD_COUNT; ++i) {
        if (output.before[i] != 0x35 ||
            output.after[i] != 0x6a) {
            printf(
                "FAIL case %u: %s guard changed\n",
                passed_cases, name
            );
            return 1;
        }
    }

    printf(
        "Case %u: %s PASS (%s)\n",
        passed_cases, name,
        cause != 0 ? "expected trap" : "completed"
    );

    ++passed_cases;

    if (cause != 0) {
        ++expected_traps;
    }

    return 0;
}

#define CHECK_RUN(...)                   \
    do {                                 \
        if (check_run(__VA_ARGS__)) {     \
            return 1;                    \
        }                                \
    } while (0)

int main(void)
{
    /*
     * Run in a fresh Spike process so all valid flags
     * initially remain false.
     */
    CHECK_RUN(
        "missing CFG",
        ILLEGAL_INSTRUCTION, UNWRITTEN, UNWRITTEN
    );

    set_cfg(1, 1);

    CHECK_RUN(
        "missing IO",
        ILLEGAL_INSTRUCTION, UNWRITTEN, UNWRITTEN
    );

    set_io();

    CHECK_RUN(
        "missing multiplier",
        ILLEGAL_INSTRUCTION, UNWRITTEN, UNWRITTEN
    );

    set_mul(INT64_C(1));

    CHECK_RUN(
        "missing zero points",
        ILLEGAL_INSTRUCTION, UNWRITTEN, UNWRITTEN
    );

    set_zp(0, 0);

    /* (2 * 1 + 1) >> 1 = 1. */
    CHECK_RUN("valid setup", 0, 1, UNWRITTEN);

    set_zp(1, 0);

    CHECK_RUN(
        "nonzero input zero point",
        ILLEGAL_INSTRUCTION, UNWRITTEN, UNWRITTEN
    );

    set_zp(0, 0);
    CHECK_RUN("recovery after invalid zero point", 0, 1, UNWRITTEN);

    input[0] = 2;
    set_mul(INT64_MAX);

    CHECK_RUN(
        "positive product overflow",
        XREQUANT_OVERFLOW, UNWRITTEN, UNWRITTEN
    );

    input[0] = -2;

    CHECK_RUN(
        "negative product overflow",
        XREQUANT_OVERFLOW, UNWRITTEN, UNWRITTEN
    );

    input[0] = 1;

    /*
     * Product = INT64_MAX fits.
     * Adding the rounding offset of 1 overflows.
     */
    CHECK_RUN(
        "rounding addition overflow",
        XREQUANT_OVERFLOW, UNWRITTEN, UNWRITTEN
    );

    /*
     * Product = INT64_MIN is valid.
     * INT64_MIN + 1 fits; shifting and saturation produce -128.
     */
    set_mul(INT64_MIN);
    CHECK_RUN("valid INT64_MIN product", 0, -128, UNWRITTEN);

    /*
     * First element succeeds and stores 0.
     * Second element's multiplication overflows.
     */
    input[0] = 0;
    input[1] = 2;
    set_cfg(2, 1);
    set_mul(INT64_MAX);

    CHECK_RUN(
        "earlier store survives later overflow",
        XREQUANT_OVERFLOW, 0, UNWRITTEN
    );

    /* Correct the source and rerun using retained setup. */
    input[1] = 0;
    CHECK_RUN("full rerun after correcting input", 0, 0, 0);

    printf(
        "PASS: %u cases, %u expected traps; "
        "output and guard checks passed\n",
        passed_cases, expected_traps
    );

    return 0;
}

#undef CHECK_RUN