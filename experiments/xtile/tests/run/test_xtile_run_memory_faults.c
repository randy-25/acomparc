#include <stdint.h>
#include <stdio.h>

#define RAM_END UINT64_C(0x80100000)
#define TAIL_GUARD_ADDRESS (RAM_END - UINT64_C(32))
#define TAIL_GUARD UINT64_C(0x123456789abcdef0)

#define UNWRITTEN32 INT32_C(0x55555555)
#define UNWRITTEN64 INT64_C(0x5555555555555555)

enum {
    LOAD_ACCESS_FAULT = 5,
    STORE_ACCESS_FAULT = 7,
    OUTPUT_CAPACITY = 4
};

extern unsigned char _end[];
extern void xtile_run_test_execute(void);
extern const unsigned char xtile_run_test_insn[];

extern void tohost_exit(uintptr_t code)
    __attribute__((noreturn));

static volatile unsigned int trap_expected;
static volatile unsigned int trap_seen;
static volatile uintptr_t expected_cause;

static unsigned int passed_cases;
static unsigned int expected_traps;

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
     * Every intentional fault in this test accesses RAM_END.
     * Memory traps report the faulting address in mtval.
     */
    if (!trap_expected ||
        trap_seen ||
        cause != expected_cause ||
        epc != (uintptr_t)xtile_run_test_insn ||
        tval != (uintptr_t)RAM_END) {

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
    unsigned int m,
    unsigned int n,
    unsigned int k,
    unsigned int tile_rows,
    unsigned int tile_columns
)
{
    const uint64_t cfg =
        (uint64_t)mode |
        ((uint64_t)(tile_rows - 1u) << 2) |
        ((uint64_t)(tile_columns - 1u) << 5) |
        ((uint64_t)m << 8) |
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
 * expected32/expected64:
 *   NULL means that entire normal buffer must stay unwritten.
 *
 * tail_width:
 *   0: no tail-output check.
 *   4: check two INT32 results immediately below RAM_END.
 *   8: check two INT64 results immediately below RAM_END.
 */
static void check_run(
    const char *name,
    uintptr_t cause,
    const int32_t *expected32,
    const int64_t *expected64,
    unsigned int tail_width
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

    __asm__ volatile("" : : : "memory");
    xtile_run_test_execute();
    __asm__ volatile("" : : : "memory");

    trap_expected = 0;

    if (cause != 0 && !trap_seen) {
        printf("Case %u FAIL: %s did not trap\n",
               passed_cases, name);
        tohost_exit(1);
    }

    for (unsigned int i = 0; i < OUTPUT_CAPACITY; ++i) {
        const int32_t wanted32 =
            expected32 ? expected32[i] : UNWRITTEN32;

        const int64_t wanted64 =
            expected64 ? expected64[i] : UNWRITTEN64;

        if (output32.values[i] != wanted32 ||
            output64.values[i] != wanted64) {
            printf("Case %u FAIL: %s, normal output slot %u\n",
                   passed_cases, name, i);
            tohost_exit(1);
        }
    }

    if (output32.before != before ||
        output32.after != after ||
        output64.before != before ||
        output64.after != after) {
        printf("Case %u FAIL: normal-output sentinel changed\n",
               passed_cases);
        tohost_exit(1);
    }

    if (tail_width != 0) {
        const uintptr_t base =
            (uintptr_t)(RAM_END - 2u * tail_width);

        const int64_t wanted[2] = {6, 8};

        for (unsigned int i = 0; i < 2; ++i) {
            int64_t actual;

            if (tail_width == 4) {
                actual = ((volatile int32_t *)base)[i];
            } else {
                actual = ((volatile int64_t *)base)[i];
            }

            if (actual != wanted[i]) {
                printf("Case %u FAIL: retained tail output %u\n",
                       passed_cases, i);
                tohost_exit(1);
            }
        }

        if (*(volatile uint64_t *)(uintptr_t)TAIL_GUARD_ADDRESS
                != TAIL_GUARD) {
            printf("Case %u FAIL: tail-output guard changed\n",
                   passed_cases);
            tohost_exit(1);
        }
    }

    printf("Case %u: %s PASS (%s)\n",
           passed_cases, name,
           cause != 0 ? "expected trap" : "completed");

    ++passed_cases;

    if (cause != 0) {
        ++expected_traps;
    }
}

static const int8_t a8 = 2;
static const int8_t bt8[4] = {3, 4, 5, 6};

static const int32_t a32 = 2;
static const int32_t bt32[4] = {3, 4, 5, 6};

static const int32_t expected_single32[4] = {
    6, UNWRITTEN32, UNWRITTEN32, UNWRITTEN32
};

static const int32_t expected_late_bt32[4] = {
    6, 8, UNWRITTEN32, UNWRITTEN32
};

static const int32_t expected_late_a32[4] = {
    6, 9, UNWRITTEN32, UNWRITTEN32
};

static const int64_t expected_complete64[4] = {
    6, 8, 10, 12
};

int main(void)
{
    /*
     * The provided crt.S places a 128 KiB stack above _end.
     * Reserve additional space and keep the tail scratch separate.
     */
    if ((uintptr_t)_end + UINT64_C(0x20000) + UINT64_C(4096)
            >= (uintptr_t)TAIL_GUARD_ADDRESS) {
        printf("FAIL: program and startup stack too close to RAM tail\n");
        return 1;
    }

    printf("xtile.run memory-fault tests: requires Spike -m1\n");

    const void *bad_input = (const void *)(uintptr_t)RAM_END;
    void *bad_output = (void *)(uintptr_t)RAM_END;

    set_cfg(0, 1, 1, 1, 1, 1);
    set_c(output32.values);
    set_zp(0, 0);

    set_ab(bad_input, bt8);
    check_run("unmapped INT8 A",
              LOAD_ACCESS_FAULT, NULL, NULL, 0);

    set_ab(&a8, bad_input);
    check_run("unmapped INT8 BT",
              LOAD_ACCESS_FAULT, NULL, NULL, 0);

    set_ab(&a8, bt8);
    set_c(bad_output);
    check_run("unmapped INT32 output",
              STORE_ACCESS_FAULT, NULL, NULL, 0);

    set_cfg(3, 1, 1, 1, 1, 1);
    set_c(output64.values);

    set_ab(bad_input, bt32);
    check_run("unmapped INT32 A",
              LOAD_ACCESS_FAULT, NULL, NULL, 0);

    set_ab(&a32, bad_input);
    check_run("unmapped INT32 BT",
              LOAD_ACCESS_FAULT, NULL, NULL, 0);

    set_ab(&a32, bt32);
    set_c(bad_output);
    check_run("unmapped INT64 output",
              STORE_ACCESS_FAULT, NULL, NULL, 0);

    set_cfg(0, 1, 1, 1, 1, 1);
    set_ab(&a8, bt8);
    set_c(output32.values);
    check_run("recovery with valid addresses",
              0, expected_single32, NULL, 0);

    /*
     * BT has only two accessible bytes before RAM_END.
     * With 1x1 output tiles, C[0] and C[1] are stored before
     * loading BT[2] faults.
     */
    volatile int8_t *tail_bytes =
        (volatile int8_t *)(uintptr_t)(RAM_END - 2);

    tail_bytes[0] = 3;
    tail_bytes[1] = 4;

    set_cfg(0, 1, 4, 1, 1, 1);
    set_ab(&a8, (const void *)(uintptr_t)(RAM_END - 2));
    check_run("earlier tiles retained after BT load fault",
              LOAD_ACCESS_FAULT, expected_late_bt32, NULL, 0);

    /*
     * A has two accessible rows, each containing one INT8.
     * Results are 2*3=6 and 3*3=9 before the next row faults.
     */
    tail_bytes[0] = 2;
    tail_bytes[1] = 3;

    set_cfg(0, 4, 1, 1, 1, 1);
    set_ab((const void *)(uintptr_t)(RAM_END - 2), bt8);
    check_run("earlier tiles retained after A load fault",
              LOAD_ACCESS_FAULT, expected_late_a32, NULL, 0);

    /*
     * One 1x4 output tile.
     * All arithmetic succeeds, but only two INT32 stores fit.
     */
    *(volatile uint64_t *)(uintptr_t)TAIL_GUARD_ADDRESS = TAIL_GUARD;

    volatile int32_t *tail32 =
        (volatile int32_t *)(uintptr_t)(RAM_END - 8);

    tail32[0] = UNWRITTEN32;
    tail32[1] = UNWRITTEN32;

    set_cfg(0, 1, 4, 1, 1, 4);
    set_ab(&a8, bt8);
    set_c((void *)(uintptr_t)(RAM_END - 8));
    check_run("partial current tile: INT32 store fault",
              STORE_ACCESS_FAULT, NULL, NULL, 4);

    /*
     * Same operation with INT32 inputs and INT64 output.
     * Two eight-byte stores fit; the third faults at RAM_END.
     */
    *(volatile uint64_t *)(uintptr_t)TAIL_GUARD_ADDRESS = TAIL_GUARD;

    volatile int64_t *tail64 =
        (volatile int64_t *)(uintptr_t)(RAM_END - 16);

    tail64[0] = UNWRITTEN64;
    tail64[1] = UNWRITTEN64;

    set_cfg(3, 1, 4, 1, 1, 4);
    set_ab(&a32, bt32);
    set_c((void *)(uintptr_t)(RAM_END - 16));
    check_run("partial current tile: INT64 store fault",
              STORE_ACCESS_FAULT, NULL, NULL, 8);

    /* Correct only C; reuse all other retained setup. */
    set_c(output64.values);
    check_run("complete rerun after correcting C",
              0, NULL, expected_complete64, 0);

    printf("PASS: %u cases, %u expected memory-access traps; "
           "output checks passed\n",
           passed_cases, expected_traps);

    return 0;
}