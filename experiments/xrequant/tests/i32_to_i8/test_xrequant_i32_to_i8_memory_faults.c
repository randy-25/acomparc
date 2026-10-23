#include <stdint.h>
#include <stdio.h>

#define RAM_END UINT64_C(0x80100000)
#define TAIL_GUARD_ADDRESS (RAM_END - UINT64_C(16))
#define TAIL_GUARD UINT64_C(0x123456789abcdef0)

enum {
    LOAD_ACCESS_FAULT = 5,
    STORE_ACCESS_FAULT = 7,
    COUNT = 4,
    GUARD_COUNT = 8,
    UNWRITTEN = 0x55
};

extern unsigned char _end[];
extern void xrequant_run_test_execute(void);
extern const unsigned char xrequant_run_test_insn[];

extern void tohost_exit(uintptr_t code)
    __attribute__((noreturn));

static volatile unsigned int trap_expected;
static volatile unsigned int trap_seen;
static volatile uintptr_t expected_cause;

static unsigned int passed_cases;
static unsigned int expected_traps;

static const int32_t input[COUNT] = {12, 16, 20, 24};

static const int8_t expected_full[COUNT] = {6, 8, 10, 12};

static const int8_t expected_partial[COUNT] = {
    6, 8, UNWRITTEN, UNWRITTEN
};

static struct {
    int8_t before[GUARD_COUNT];
    int8_t values[COUNT];
    int8_t after[GUARD_COUNT];
} output;

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
     * Every deliberate memory fault accesses RAM_END.
     * mtval must contain the address, not instruction bits.
     */
    if (!trap_expected ||
        trap_seen ||
        cause != expected_cause ||
        epc != (uintptr_t)xrequant_run_test_insn ||
        tval != (uintptr_t)RAM_END) {

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

static void set_io(const void *src, void *dst)
{
    __asm__ volatile(
        ".insn r 0x2b, 1, 0, x0, %0, %1"
        : : "r"((uintptr_t)src), "r"((uintptr_t)dst)
    );
}

static void setup(unsigned int count, const void *src, void *dst)
{
    /* MODE=2, SHIFT=1, multiplier=1, both zero points=0. */
    const uint64_t cfg =
        UINT64_C(2) |
        (UINT64_C(1) << 3) |
        ((uint64_t)count << 9);

    const int64_t multiplier = 1;

    __asm__ volatile(
        ".insn r 0x2b, 0, 0, x0, %0, x0\n\t"
        ".insn r 0x2b, 1, 0, x0, %1, %2\n\t"
        ".insn r 0x2b, 2, 0, x0, %3, x0\n\t"
        ".insn r 0x2b, 3, 0, x0, x0, x0"
        :
        : "r"(cfg),
          "r"((uintptr_t)src),
          "r"((uintptr_t)dst),
          "r"(multiplier)
        : "memory"
    );
}

static int check_run(
    const char *name,
    uintptr_t cause,
    const int8_t *expected,
    int check_tail
)
{
    for (unsigned int i = 0; i < GUARD_COUNT; ++i) {
        output.before[i] = 0x35;
        output.after[i] = 0x6a;
    }

    for (unsigned int i = 0; i < COUNT; ++i) {
        output.values[i] = UNWRITTEN;
    }

    trap_seen = 0;
    expected_cause = cause;
    trap_expected = cause != 0;

    __asm__ volatile("" : : : "memory");
    xrequant_run_test_execute();
    __asm__ volatile("" : : : "memory");

    trap_expected = 0;

    if (cause != 0 && !trap_seen) {
        printf("FAIL case %u: %s did not trap\n", passed_cases, name);
        return 1;
    }

    for (unsigned int i = 0; i < COUNT; ++i) {
        const int wanted = expected ? expected[i] : UNWRITTEN;

        if (output.values[i] != wanted) {
            printf(
                "FAIL case %u: %s output %u got=%d expected=%d\n",
                passed_cases, name, i,
                (int)output.values[i], wanted
            );
            return 1;
        }
    }

    for (unsigned int i = 0; i < GUARD_COUNT; ++i) {
        if (output.before[i] != 0x35 ||
            output.after[i] != 0x6a) {
            printf("FAIL case %u: normal guard changed\n", passed_cases);
            return 1;
        }
    }

    if (check_tail) {
        volatile int8_t *tail =
            (volatile int8_t *)(uintptr_t)(RAM_END - 2);

        if (tail[0] != 6 || tail[1] != 8 ||
            *(volatile uint64_t *)(uintptr_t)TAIL_GUARD_ADDRESS
                != TAIL_GUARD) {
            printf(
                "FAIL case %u: tail output or guard mismatch\n",
                passed_cases
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
    expected_traps += cause != 0;
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
     * Existing CV-Wally startup reserves a 128 KiB stack above _end.
     * Keep the deliberate RAM-tail scratch region separate.
     */
    if ((uintptr_t)_end + UINT64_C(0x20000) + UINT64_C(4096)
            >= (uintptr_t)TAIL_GUARD_ADDRESS) {
        printf("FAIL: program/startup stack too close to RAM tail\n");
        return 1;
    }

    printf("xrequant memory-fault test: requires Spike -m1\n");

    const void *bad_src = (const void *)(uintptr_t)RAM_END;
    void *bad_dst = (void *)(uintptr_t)RAM_END;

    setup(1, bad_src, output.values);
    CHECK_RUN("unmapped source", LOAD_ACCESS_FAULT, NULL, 0);

    setup(1, input, bad_dst);
    CHECK_RUN("unmapped destination", STORE_ACCESS_FAULT, NULL, 0);

    setup(COUNT, input, output.values);
    CHECK_RUN("valid address recovery", 0, expected_full, 0);

    /*
     * Only two INT32 elements remain before RAM_END.
     * The third source load must fault after storing outputs 6 and 8.
     */
    volatile int32_t *tail_source =
        (volatile int32_t *)(uintptr_t)(RAM_END - 8);

    tail_source[0] = 12;
    tail_source[1] = 16;

    setup(
        COUNT,
        (const void *)(uintptr_t)(RAM_END - 8),
        output.values
    );

    CHECK_RUN(
        "earlier outputs survive source fault",
        LOAD_ACCESS_FAULT, expected_partial, 0
    );

    /*
     * Only two destination bytes remain before RAM_END.
     * The third store faults after storing outputs 6 and 8.
     */
    volatile int8_t *tail_destination =
        (volatile int8_t *)(uintptr_t)(RAM_END - 2);

    tail_destination[0] = UNWRITTEN;
    tail_destination[1] = UNWRITTEN;

    *(volatile uint64_t *)(uintptr_t)TAIL_GUARD_ADDRESS = TAIL_GUARD;

    setup(COUNT, input, (void *)(uintptr_t)(RAM_END - 2));

    CHECK_RUN(
        "earlier outputs survive destination fault",
        STORE_ACCESS_FAULT, NULL, 1
    );

    /* Keep CFG, multiplier, and zero points; correct the destination. */
    set_io(input, output.values);

    CHECK_RUN("full rerun after correcting destination", 0, expected_full, 0);

    printf(
        "PASS: %u cases, %u expected memory-access traps; "
        "output checks passed\n",
        passed_cases, expected_traps
    );

    return 0;
}

#undef CHECK_RUN