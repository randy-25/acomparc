#include <stdint.h>
#include <stdio.h>

#define OUTPUT_CAPACITY 4
#define UNWRITTEN32 INT32_C(0x55555555)
#define UNWRITTEN64 INT64_C(0x5555555555555555)

enum {
    ILLEGAL_INSTRUCTION = 2,
    XTILE_OVERFLOW = 24
};

extern void xtile_run_test_execute(void);
extern const unsigned char xtile_run_test_insn[];

extern void tohost_exit(uintptr_t code)
    __attribute__((noreturn));

static volatile unsigned int trap_expected;
static volatile unsigned int trap_seen;
static volatile uintptr_t expected_cause;

static unsigned int passed_cases;
static unsigned int expected_traps;

/*
 * Strong definition overrides the common runtime's weak handler.
 * Only the expected RUN exception is allowed to resume.
 */
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
        epc != (uintptr_t)xtile_run_test_insn ||
        tval != UINT32_C(0x0000400b)) {

        printf("FAIL: unexpected trap: cause=%u "
               "epc=%llx tval=%llx\n",
               (unsigned int)cause,
               (unsigned long long)epc,
               (unsigned long long)tval);

        tohost_exit(1);
    }

    trap_seen = 1;
    return epc + 4;
}

static inline void set_cfg(
    unsigned int mode,
    unsigned int n,
    unsigned int k
)
{
    /*
     * M=1 and active tile=1x1.
     * This makes output tiles execute one element at a time.
     */
    const uint64_t cfg =
        (uint64_t)mode |
        (UINT64_C(1) << 8) |
        ((uint64_t)n << 18) |
        ((uint64_t)k << 28);

    __asm__ volatile(
        ".insn r 0x0b, 0, 0, x0, %0, x0"
        :
        : "r"(cfg)
    );
}

static inline void set_ab(const void *a, const void *bt)
{
    __asm__ volatile(
        ".insn r 0x0b, 1, 0, x0, %0, %1"
        :
        : "r"((uintptr_t)a), "r"((uintptr_t)bt)
    );
}

static inline void set_c(void *c)
{
    __asm__ volatile(
        ".insn r 0x0b, 2, 0, x0, %0, x0"
        :
        : "r"((uintptr_t)c)
    );
}

static inline void set_zp(int32_t a_zp, int32_t b_zp)
{
    __asm__ volatile(
        ".insn r 0x0b, 3, 0, x0, %0, %1"
        :
        : "r"((intptr_t)a_zp), "r"((intptr_t)b_zp)
    );
}

static struct {
    uint64_t before;
    int32_t values[OUTPUT_CAPACITY];
    uint64_t after;
} output32;

static struct {
    uint64_t before;
    int64_t values[OUTPUT_CAPACITY];
    uint64_t after;
} output64;

/*
 * cause=0 means successful execution is expected.
 *
 * expected_count=0 means the entire selected buffer must remain
 * unwritten. Otherwise compare the first expected_count elements
 * and require all remaining slots to remain unwritten.
 */
static int check_run(
    const char *name,
    uintptr_t cause,
    int wide,
    const int64_t *expected,
    unsigned int expected_count
)
{
    const uint64_t before = UINT64_C(0x0123456789abcdef);
    const uint64_t after = UINT64_C(0xfedcba9876543210);

    output32.before = before;
    output32.after = after;
    output64.before = before;
    output64.after = after;

    for (unsigned int i = 0; i < OUTPUT_CAPACITY; ++i) {
        output32.values[i] = UNWRITTEN32;
        output64.values[i] = UNWRITTEN64;
    }

    trap_seen = 0;
    expected_cause = cause;
    trap_expected = cause != 0;

    /* Compiler barriers around the external assembly helper. */
    __asm__ volatile("" : : : "memory");
    xtile_run_test_execute();
    __asm__ volatile("" : : : "memory");

    trap_expected = 0;

    if (cause != 0 && !trap_seen) {
        printf("Case %u FAIL: %s did not trap\n",
               passed_cases, name);
        return 1;
    }

    for (unsigned int i = 0; i < OUTPUT_CAPACITY; ++i) {
        const int64_t actual =
            wide ? output64.values[i] : (int64_t)output32.values[i];

        const int64_t wanted =
            i < expected_count
                ? expected[i]
                : (wide ? UNWRITTEN64 : (int64_t)UNWRITTEN32);

        if (actual != wanted) {
            printf("Case %u FAIL: %s, output %u\n",
                   passed_cases, name, i);

            printf("actual=%lld expected=%lld\n",
                   (long long)actual,
                   (long long)wanted);
            return 1;
        }
    }

    if (output32.before != before ||
        output32.after != after ||
        output64.before != before ||
        output64.after != after) {
        printf("Case %u FAIL: sentinel changed\n", passed_cases);
        return 1;
    }

    printf("Case %u: %s PASS (%s)\n",
           passed_cases,
           name,
           cause != 0 ? "expected trap" : "completed");

    ++passed_cases;

    if (cause != 0) {
        ++expected_traps;
    }

    return 0;
}

static const int8_t basic_a = 2;
static const int8_t basic_bt = 3;
static const int64_t basic_expected[] = {6};

static const int32_t i32_high = INT32_MAX;
static const int32_t i32_low = INT32_MIN;

static const int32_t low_pair[] = {
    INT32_MIN, INT32_MIN
};

static const int32_t low_triplet[] = {
    INT32_MIN, INT32_MIN, INT32_MIN
};

static const int32_t high_triplet[] = {
    INT32_MAX, INT32_MAX, INT32_MAX
};

/*
 * A = [INT32_MIN, INT32_MIN]
 *
 * First output: INT32_MIN*1 + INT32_MIN*0.
 * Second output: 2 * INT32_MIN*INT32_MIN, which overflows INT64.
 */
static const int32_t partial_bt[] = {
    1, 0,
    INT32_MIN, INT32_MIN
};

static const int64_t partial_expected[] = {
    -INT64_C(2147483648),
    UNWRITTEN64
};

/* Replacement BT permits the entire operation to complete. */
static const int32_t corrected_bt[] = {
    1, 0,
    1, 1
};

static const int64_t corrected_expected[] = {
    -INT64_C(2147483648),
    -INT64_C(4294967296)
};

int main(void)
{
    printf("xtile.run setup and arithmetic exception tests\n");

    /*
     * This ELF starts in a fresh Spike process.
     * Supply setup incrementally to exercise each missing flag.
     */

    if (check_run("missing configuration",
                  ILLEGAL_INSTRUCTION, 0, NULL, 0)) {
        return 1;
    }

    set_cfg(0, 1, 1);

    if (check_run("missing A/BT addresses",
                  ILLEGAL_INSTRUCTION, 0, NULL, 0)) {
        return 1;
    }

    set_ab(&basic_a, &basic_bt);

    if (check_run("missing C address",
                  ILLEGAL_INSTRUCTION, 0, NULL, 0)) {
        return 1;
    }

    set_c(output32.values);

    if (check_run("missing zero-point initialization",
                  ILLEGAL_INSTRUCTION, 0, NULL, 0)) {
        return 1;
    }

    set_zp(0, 0);

    if (check_run("all setup supplied",
                  0, 0, basic_expected, 1)) {
        return 1;
    }

    static const struct {
        const char *name;
        int32_t a_zp;
        int32_t b_zp;
    } invalid_zp[] = {
        {"reject INT8 A zero point +128",  128,    0},
        {"reject INT8 A zero point -129", -129,    0},
        {"reject INT8 B zero point +128",    0,  128},
        {"reject INT8 B zero point -129",    0, -129}
    };

    for (unsigned int i = 0;
         i < (unsigned int)(sizeof(invalid_zp) / sizeof(invalid_zp[0]));
         ++i) {

        set_zp(invalid_zp[i].a_zp, invalid_zp[i].b_zp);

        if (check_run(invalid_zp[i].name,
                      ILLEGAL_INSTRUCTION, 0, NULL, 0)) {
            return 1;
        }
    }

    set_zp(0, 0);

    if (check_run("recovery after invalid zero points",
                  0, 0, basic_expected, 1)) {
        return 1;
    }

    /* Positive product exceeds INT64_MAX. */
    set_cfg(3, 1, 1);
    set_ab(&i32_high, &i32_high);
    set_c(output64.values);
    set_zp(INT32_MIN, INT32_MIN);

    if (check_run("positive multiplication overflow",
                  XTILE_OVERFLOW, 1, NULL, 0)) {
        return 1;
    }

    /* Negative product is below INT64_MIN. */
    set_ab(&i32_high, &i32_low);
    set_zp(INT32_MIN, INT32_MAX);

    if (check_run("negative multiplication overflow",
                  XTILE_OVERFLOW, 1, NULL, 0)) {
        return 1;
    }

    /*
     * Each product is 2^62 and fits INT64.
     * Adding the second product produces 2^63.
     */
    set_cfg(3, 1, 2);
    set_ab(low_pair, low_pair);
    set_zp(0, 0);

    if (check_run("positive accumulation overflow",
                  XTILE_OVERFLOW, 1, NULL, 0)) {
        return 1;
    }

    /*
     * Each product is -2^62.
     * The second partial sum equals INT64_MIN.
     * The third partial sum overflows.
     */
    set_cfg(3, 1, 3);
    set_ab(low_triplet, high_triplet);
    set_zp(0, -1);

    if (check_run("negative accumulation overflow",
                  XTILE_OVERFLOW, 1, NULL, 0)) {
        return 1;
    }

    /*
     * M=1, N=2, K=2, tile=1x1.
     * Output tile 0 completes; output tile 1 overflows.
     */
    set_cfg(3, 2, 2);
    set_ab(low_pair, partial_bt);
    set_zp(0, 0);

    if (check_run("earlier tile retained after later overflow",
                  XTILE_OVERFLOW, 1, partial_expected, 2)) {
        return 1;
    }

    /*
     * Replace only the input addresses.
     * Retained cfg, C address, and zero points are reused.
     */
    set_ab(low_pair, corrected_bt);

    if (check_run("complete rerun after correcting inputs",
                  0, 1, corrected_expected, 2)) {
        return 1;
    }

    printf("PASS: %u cases, %u expected traps; "
           "output and sentinel checks passed\n",
           passed_cases, expected_traps);

    return 0;
}