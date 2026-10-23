#include <stdint.h>
#include <stdio.h>

#include "requantization.h"

#define CAPACITY 257
#define UNWRITTEN 0x55

static int32_t input[CAPACITY];
static int32_t input_copy[CAPACITY];
static int8_t reference[CAPACITY];

static struct {
    int8_t before;
    int8_t values[CAPACITY];
    int8_t after;
} output;

static unsigned int completed;
static unsigned int checked_outputs;
static uint32_t random_state = UINT32_C(0x13579bdf);

static uint32_t next_random(void)
{
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

static void prepare_buffers(void)
{
    static const int32_t anchors[] = {
        INT32_MIN, INT32_MAX,
        -10, 10, -5, 5, -1, 1, 0, -6, 6
    };

    for (unsigned int i = 0; i < CAPACITY; ++i) {
        input[i] = (int32_t)next_random();
        output.values[i] = UNWRITTEN;
    }

    for (size_t i = 0; i < sizeof anchors / sizeof anchors[0]; ++i) {
        input[i] = anchors[i];
    }

    output.before = 0x35;
    output.after = 0x6a;
}

static void run_mode4(
    size_t count,
    int32_t multiplier,
    unsigned int second_shift,
    int32_t output_zp
)
{
    const uint64_t cfg =
        UINT64_C(4) |
        ((uint64_t)second_shift << 3) |
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
          "r"((int64_t)multiplier),
          "r"((intptr_t)output_zp)
        : "memory"
    );
}

static int execute_case(
    size_t count,
    int32_t multiplier,
    unsigned int second_shift,
    int32_t output_zp
)
{
    for (unsigned int i = 0; i < CAPACITY; ++i) {
        input_copy[i] = input[i];
    }

    run_mode4(count, multiplier, second_shift, output_zp);

    for (size_t i = 0; i < CAPACITY; ++i) {
        const int wanted = i < count ? reference[i] : UNWRITTEN;

        if (output.values[i] != wanted) {
            printf(
                "FAIL case %u index=%u got=%d expected=%d\n",
                completed, (unsigned int)i,
                (int)output.values[i], wanted
            );

            printf(
                "Parameters: count=%u multiplier=%d shift=%u zp=%d\n",
                (unsigned int)count, (int)multiplier,
                second_shift, (int)output_zp
            );

            return 1;
        }

        if (input[i] != input_copy[i]) {
            printf(
                "FAIL case %u: input index %u changed\n",
                completed, (unsigned int)i
            );
            return 1;
        }
    }

    if (output.before != 0x35 || output.after != 0x6a) {
        printf("FAIL case %u: output guard changed\n", completed);
        return 1;
    }

    ++completed;
    checked_outputs += (unsigned int)count;
    return 0;
}

int main(void)
{
    /*
     * Independent expected values for both rounding stages.
     * Multiplier 2^30 represents 0.5 in Q31.
     */
    static const int32_t tie_input[] = {
        -10, 10, -5, 5, -1, 1, 0, -6, 6
    };

    static const int8_t tie_expected[] = {
        -3, 3, -1, 2, 0, 1, 0, -2, 2
    };

    prepare_buffers();

    for (size_t i = 0; i < sizeof tie_input / sizeof tie_input[0]; ++i) {
        input[i] = tie_input[i];
        reference[i] = tie_expected[i];
    }

    if (execute_case(9, INT32_C(1073741824), 1, 0)) {
        return 1;
    }

    /* Second shift zero, including saturation. */
    prepare_buffers();
    input[0] = INT32_MIN;
    input[1] = INT32_MAX;
    reference[0] = -128;
    reference[1] = 127;

    if (execute_case(2, INT32_MAX, 0, 0)) {
        return 1;
    }

    /* Maximum supported second shift. */
    prepare_buffers();
    input[0] = INT32_MIN;
    input[1] = INT32_MAX;
    reference[0] = -1;
    reference[1] = 1;

    if (execute_case(2, INT32_MAX, 31, 0)) {
        return 1;
    }

    static const int32_t multipliers[] = {
        1,
        INT32_C(1073741824),
        INT32_C(1195729761),
        INT32_MAX
    };

    static const int32_t zero_points[] = {
        -128, -88, 0, 127, INT32_MIN, INT32_MAX
    };

    static const size_t counts[] = {1, 3, 32, CAPACITY};

    unsigned int differential_cases = 0;

    for (size_t m = 0;
         m < sizeof multipliers / sizeof multipliers[0]; ++m) {
        for (unsigned int shift = 0; shift <= 31; ++shift) {
            for (size_t c = 0;
                 c < sizeof counts / sizeof counts[0]; ++c) {

                const int32_t multiplier = multipliers[m];
                const size_t count = counts[c];

                int32_t zp = zero_points[
                    differential_cases %
                    (sizeof zero_points / sizeof zero_points[0])
                ];

                /* Explicitly exercise the current GELU parameters. */
                if (multiplier == INT32_C(1195729761) && shift == 6) {
                    zp = -88;
                }

                prepare_buffers();

                requantize_two_round_i32_to_i8(
                    input, reference, count,
                    multiplier, shift, zp
                );

                if (execute_case(count, multiplier, shift, zp)) {
                    return 1;
                }

                ++differential_cases;
            }
        }
    }

    printf(
        "PASS: %u cases, %u outputs; "
        "directed rounding and scalar comparisons passed; "
        "inputs, guards, and unused slots unchanged\n",
        completed, checked_outputs
    );

    return 0;
}