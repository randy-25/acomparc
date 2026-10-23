#include <stdint.h>
#include <stdio.h>

#include "requantization.h"

#define CAPACITY 257
#define UNWRITTEN32 INT32_C(0x55555555)
#define UNWRITTEN8 0x55

typedef struct {
    unsigned int mode;
    unsigned int shift;
    int64_t multiplier;
    int32_t output_zp;
} TestCase;

static const TestCase cases[] = {
    /* Mode 0 ignores the multiplier. */
    {0,  3,       1, 0},
    {0, 15,       1, 0},
    {0, 23,      -3, 0},
    {0, 32,       1, 0},
    {0, 47,       1, 0},
    {0, 63,       1, 0},

    /* Mode 1: multiply, round, shift, zero point, saturation. */
    {1,  1,          3,  -5},
    {1,  3,         -3,  10},
    {1, 39,     974600, -17},
    {1, 47,    1009162,   2},
    {1, 63, 1073741823,   0},
    {1, 15,          0, 127},
    {1, 32,          1,   0},
    {1, 63,          1,   0}
};

static int64_t input[CAPACITY];
static int32_t reference32[CAPACITY];
static int8_t reference8[CAPACITY];

static struct {
    uint32_t before;
    int32_t values[CAPACITY];
    uint32_t after;
} output32;

static struct {
    int8_t before;
    int8_t values[CAPACITY];
    int8_t after;
} output8;

static uint32_t random_state = UINT32_C(0x2468ace1);

static uint32_t next_random(void)
{
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

static void prepare_input(const TestCase *p)
{
    const int64_t rounding =
        INT64_C(1) << (p->shift - 1);

    for (unsigned int i = 0; i < CAPACITY; ++i) {
        if (p->mode == 0 && p->shift < 32) {
            /*
             * Construct values whose rounded results fit INT32,
             * while allowing inputs larger than INT32.
             */
            const int64_t whole =
                (int64_t)(next_random() & UINT32_C(0x1fffff))
                - INT64_C(1048576);

            const int64_t scale = INT64_C(1) << p->shift;
            const uint64_t mask =
                (UINT64_C(1) << p->shift) - 1;

            const int64_t fraction =
                (int64_t)((uint64_t)next_random() & mask);

            input[i] = whole * scale + fraction;
        } else {
            /*
             * Magnitude <= UINT32_MAX.
             * For the listed multipliers, product + rounding fits INT64.
             */
            const int64_t magnitude = (int64_t)next_random();
            input[i] = (i & 1u) ? -magnitude : magnitude;
        }
    }

    if (p->mode == 0) {
        if (p->shift >= 32) {
            /*
             * Valid extreme INT64 inputs.
             * The positive value leaves room for the rounding offset.
             */
            input[0] = INT64_MIN;
            input[1] = INT64_MAX - rounding;
        } else {
            /* Positive/negative exact halfway cases. */
            input[0] = rounding;
            input[1] = -rounding;
        }
    } else {
        input[0] = INT64_C(4294967295);
        input[1] = -INT64_C(4294967295);

        if (p->multiplier == 1) {
            input[0] = INT64_MIN;
            input[1] = INT64_MAX - rounding;
        }
    }

    input[2] = 0;
}

static void prepare_output(void)
{
    output32.before = UINT32_C(0x13579bdf);
    output32.after = UINT32_C(0x2468ace0);
    output8.before = 0x35;
    output8.after = 0x6a;

    for (unsigned int i = 0; i < CAPACITY; ++i) {
        output32.values[i] = UNWRITTEN32;
        output8.values[i] = UNWRITTEN8;
    }
}

static void run_xrequant(const TestCase *p, size_t count)
{
    const uint64_t cfg =
        (uint64_t)p->mode |
        ((uint64_t)p->shift << 3) |
        ((uint64_t)count << 9);

    void *destination =
        p->mode == 0
            ? (void *)output32.values
            : (void *)output8.values;

    __asm__ volatile(
        ".insn r 0x2b, 0, 0, x0, %0, x0\n\t"
        ".insn r 0x2b, 1, 0, x0, %1, %2\n\t"
        ".insn r 0x2b, 2, 0, x0, %3, x0\n\t"
        ".insn r 0x2b, 3, 0, x0, x0, %4\n\t"
        ".insn r 0x2b, 4, 0, x0, x0, x0"
        :
        : "r"(cfg),
          "r"((uintptr_t)input),
          "r"((uintptr_t)destination),
          "r"(p->multiplier),
          "r"((intptr_t)p->output_zp)
        : "memory"
    );
}

static int check_output(
    unsigned int case_id,
    const TestCase *p,
    size_t count
)
{
    /*
     * Check active output, unused slots, and the entire inactive
     * output buffer. This also catches incorrect store widths.
     */
    for (size_t i = 0; i < CAPACITY; ++i) {
        const int32_t wanted32 =
            p->mode == 0 && i < count
                ? reference32[i] : UNWRITTEN32;

        const int wanted8 =
            p->mode == 1 && i < count
                ? reference8[i] : UNWRITTEN8;

        if (output32.values[i] != wanted32 ||
            output8.values[i] != wanted8) {
            printf(
                "FAIL case %u mode=%u shift=%u index=%u\n",
                case_id, p->mode, p->shift, (unsigned int)i
            );
            printf(
                "INT32 got=%d expected=%d; INT8 got=%d expected=%d\n",
                (int)output32.values[i], (int)wanted32,
                (int)output8.values[i], wanted8
            );
            return 1;
        }
    }

    if (output32.before != UINT32_C(0x13579bdf) ||
        output32.after != UINT32_C(0x2468ace0) ||
        output8.before != 0x35 ||
        output8.after != 0x6a) {
        printf("FAIL case %u: output guard changed\n", case_id);
        return 1;
    }

    return 0;
}

int main(void)
{
    static const size_t counts[] = {1, 3, 32, CAPACITY};

    unsigned int completed = 0;
    unsigned int outputs = 0;

    for (size_t t = 0; t < sizeof cases / sizeof cases[0]; ++t) {
        const TestCase *p = &cases[t];

        for (size_t c = 0; c < sizeof counts / sizeof counts[0]; ++c) {
            const size_t count = counts[c];

            prepare_input(p);
            prepare_output();

            if (p->mode == 0) {
                requantize_shift_i64_to_i32(
                    input, reference32, count, p->shift
                );
            } else {
                requantize_i64_to_i8(
                    input, reference8, count,
                    p->multiplier, p->shift, p->output_zp
                );
            }

            run_xrequant(p, count);

            if (check_output(completed, p, count)) {
                return 1;
            }

            ++completed;
            outputs += (unsigned int)count;
        }
    }

    printf(
        "PASS: %u cases, %u outputs match scalar references; "
        "guards, unused slots, and inactive buffers unchanged\n",
        completed, outputs
    );

    return 0;
}