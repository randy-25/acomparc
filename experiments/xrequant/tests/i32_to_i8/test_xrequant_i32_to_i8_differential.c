#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#include "requantization.h"

enum {
    CAPACITY = 257,
    GUARD_COUNT = 8
};

static int32_t input[CAPACITY];
static int8_t reference[CAPACITY];

static struct {
    int8_t before[GUARD_COUNT];
    int8_t values[CAPACITY];
    int8_t after[GUARD_COUNT];
} output;

/*
 * Fixed test arguments are validated by the test tables.
 * Input zero point is supplied as zero through x0.
 */
static void run_xrequant(
    size_t count,
    int64_t multiplier,
    unsigned int shift,
    int32_t output_zp
)
{
    const uint64_t cfg =
        UINT64_C(2) |
        ((uint64_t)shift << 3) |
        ((uint64_t)count << 9);

    __asm__ volatile(
        ".insn r 0x2b, 0, 0, x0, %0, x0\n\t"
        ".insn r 0x2b, 1, 0, x0, %1, %2\n\t"
        ".insn r 0x2b, 2, 0, x0, %3, x0\n\t"
        ".insn r 0x2b, 3, 0, x0, x0, %4\n\t"
        ".insn r 0x2b, 4, 0, x0, x0, x0"
        :
        : "r"(cfg),
          "r"((uintptr_t)input),
          "r"((uintptr_t)output.values),
          "r"(multiplier),
          "r"((intptr_t)output_zp)
        : "memory"
    );
}

static uint32_t random_state = UINT32_C(0x13579bdf);

static uint32_t next_random(void)
{
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

static void prepare_input(void)
{
    for (size_t i = 0; i < CAPACITY; ++i) {
        input[i] = (int32_t)next_random();
    }

    input[0] = INT32_MIN;
    input[1] = INT32_MAX;
    input[2] = 0;
    input[3] = 4;
    input[4] = -4;
    input[5] = 20;
    input[6] = -20;
    input[7] = 1;
    input[8] = -1;
}

static void prepare_output(void)
{
    for (size_t i = 0; i < GUARD_COUNT; ++i) {
        output.before[i] = 0x35;
        output.after[i] = 0x6a;
    }

    for (size_t i = 0; i < CAPACITY; ++i) {
        output.values[i] = 0x55;
    }
}

static int check_output(unsigned int case_id, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (output.values[i] != reference[i]) {
            printf(
                "FAIL case %u index %u: input=%d got=%d expected=%d\n",
                case_id, (unsigned int)i, (int)input[i],
                (int)output.values[i], (int)reference[i]
            );
            return 1;
        }
    }

    /* RUN must store exactly COUNT elements. */
    for (size_t i = count; i < CAPACITY; ++i) {
        if (output.values[i] != 0x55) {
            printf(
                "FAIL case %u: unused output index %u changed\n",
                case_id, (unsigned int)i
            );
            return 1;
        }
    }

    for (size_t i = 0; i < GUARD_COUNT; ++i) {
        if (output.before[i] != 0x35 ||
            output.after[i] != 0x6a) {
            printf(
                "FAIL case %u: output guard %u changed\n",
                case_id, (unsigned int)i
            );
            return 1;
        }
    }

    return 0;
}

int main(void)
{
    static const size_t counts[] = {
        1, 3, 32, CAPACITY
    };

    static const unsigned int shifts[] = {
        1, 2, 3, 7, 8, 15, 31, 39, 47, 62, 63
    };

    /*
     * These multiplier bounds ensure product + rounding offset
     * fits INT64 for every INT32 input and every tested shift.
     */
    static const int64_t multipliers[] = {
        -INT64_C(2147483647),
        -INT64_C(3),
        INT64_C(0),
        INT64_C(3),
        INT64_C(2147483647)
    };

    static const int32_t zero_points[] = {
        -128, -5, 0, 127, INT32_MIN, INT32_MAX
    };

    unsigned int cases = 0;
    unsigned int outputs = 0;

    for (size_t m = 0;
         m < sizeof multipliers / sizeof multipliers[0]; ++m) {
        for (size_t s = 0;
             s < sizeof shifts / sizeof shifts[0]; ++s) {
            for (size_t c = 0;
                 c < sizeof counts / sizeof counts[0]; ++c) {

                const size_t count = counts[c];
                const int64_t multiplier = multipliers[m];
                const unsigned int shift = shifts[s];
                const int32_t zp = zero_points[
                    cases % (sizeof zero_points / sizeof zero_points[0])
                ];

                prepare_input();
                prepare_output();

                requantize_i32_to_i8(
                    input, reference, count,
                    multiplier, shift, zp
                );

                run_xrequant(count, multiplier, shift, zp);

                if (check_output(cases, count)) {
                    printf(
                        "Parameters: count=%u multiplier=%lld "
                        "shift=%u zp=%d\n",
                        (unsigned int)count,
                        (long long)multiplier,
                        shift, (int)zp
                    );
                    return 1;
                }

                outputs += (unsigned int)count;
                ++cases;
            }
        }
    }

    printf(
        "PASS: %u cases, %u outputs match scalar reference; "
        "guards and unused output slots unchanged\n",
        cases, outputs
    );

    return 0;
}