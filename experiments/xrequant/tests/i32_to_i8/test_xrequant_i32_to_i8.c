#include <stdint.h>
#include <stdio.h>

enum {
    COUNT = 12,
    GUARD_COUNT = 8
};

static inline void xrequant_cfg(uint64_t cfg)
{
    __asm__ volatile(
        ".insn r 0x2b, 0, 0, x0, %0, x0"
        : : "r"(cfg)
    );
}

static inline void xrequant_set_io(
    const void *src,
    void *dst
)
{
    __asm__ volatile(
        ".insn r 0x2b, 1, 0, x0, %0, %1"
        : : "r"((uintptr_t)src), "r"((uintptr_t)dst)
    );
}

static inline void xrequant_set_mul(int64_t multiplier)
{
    __asm__ volatile(
        ".insn r 0x2b, 2, 0, x0, %0, x0"
        : : "r"(multiplier)
    );
}

static inline void xrequant_set_zp(
    int32_t input_zp,
    int32_t output_zp
)
{
    __asm__ volatile(
        ".insn r 0x2b, 3, 0, x0, %0, %1"
        : : "r"((intptr_t)input_zp),
            "r"((intptr_t)output_zp)
    );
}

static inline void xrequant_run(void)
{
    __asm__ volatile(
        ".insn r 0x2b, 4, 0, x0, x0, x0"
        : : : "memory"
    );
}

static const int32_t input[COUNT] = {
    20, -20, 1000, -1000,
    0, 4, -4, 1,
    -1, 352, -328, 348
};

/*
 * Expected:
 * clamp((((INT64)input * 3 + 4) >> 3) - 5, -128, 127)
 *
 * Includes:
 * - positive and negative halfway cases;
 * - upper and lower saturation;
 * - exact output limits and one value below the upper limit.
 */
static const int8_t expected[COUNT] = {
    3, -12, 127, -128,
    -5, -3, -6, -5,
    -5, 127, -128, 126
};

/* All members have byte alignment: guards directly surround output. */
static struct {
    int8_t before[GUARD_COUNT];
    int8_t values[COUNT];
    int8_t after[GUARD_COUNT];
} output;

static void prepare_output(void)
{
    for (unsigned int i = 0; i < GUARD_COUNT; ++i) {
        output.before[i] = 0x35;
        output.after[i] = 0x6a;
    }

    for (unsigned int i = 0; i < COUNT; ++i) {
        output.values[i] = 0x55;
    }
}

static int check_output(unsigned int pass)
{
    unsigned int mismatches = 0;

    for (unsigned int i = 0; i < COUNT; ++i) {
        const int actual = output.values[i];
        const int wanted = expected[i];

        printf(
            "Pass %u element %u: input=%d actual=%d expected=%d %s\n",
            pass, i, (int)input[i], actual, wanted,
            actual == wanted ? "PASS" : "FAIL"
        );

        if (actual != wanted) {
            ++mismatches;
        }
    }

    for (unsigned int i = 0; i < GUARD_COUNT; ++i) {
        if (output.before[i] != 0x35 ||
            output.after[i] != 0x6a) {
            printf(
                "FAIL: pass %u output guard %u changed\n",
                pass, i
            );
            return 1;
        }
    }

    if (mismatches != 0) {
        printf(
            "FAIL: pass %u has %u output mismatches\n",
            pass, mismatches
        );
        return 1;
    }

    return 0;
}

int main(void)
{
    /*
     * MODE  = 2: INT32 -> INT8
     * SHIFT = 3
     * COUNT = 12
     */
    const uint64_t cfg =
        UINT64_C(2) |
        (UINT64_C(3) << 3) |
        ((uint64_t)COUNT << 9);

    printf(
        "xrequant INT32 -> INT8: count=%u multiplier=3 shift=3 zp=-5\n",
        (unsigned int)COUNT
    );

    prepare_output();

    xrequant_cfg(cfg);
    xrequant_set_io(input, output.values);
    xrequant_set_mul(INT64_C(3));
    xrequant_set_zp(0, -5);
    xrequant_run();

    if (check_output(0)) {
        return 1;
    }

    /*
     * Overwrite the destination, then run again without setters.
     * This verifies retained configuration and addresses.
     */
    prepare_output();
    xrequant_run();

    if (check_output(1)) {
        return 1;
    }

    printf(
        "PASS: 2 runs, 24 INT8 outputs, "
        "0 mismatches; output guards unchanged\n"
    );

    return 0;
}