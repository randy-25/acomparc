#include <stdint.h>
#include <stdio.h>

#include "requantization.h"

#define CAPACITY 256
#define UNWRITTEN INT32_C(0x55555555)

typedef struct {
    int32_t input_zp;
    int64_t multiplier;
    unsigned int shift;
} TestCase;

static const TestCase cases[] = {
    {-128,                 1,  1},
    { -17,                 3,  3},
    {   0,                -3,  3},
    {   2,                 5,  2},
    { 127,                 1,  1},
    {   0,                 0,  1},
    {   0,                 1, 63},
    {   0, INT64_C(1) << 40, 39},

    /* Current block-0 residual branch conversions. */
    { -13,         939062400, 22},
    {  10,         720118272, 22},

    /* Verify full signed INT32 zero-point interpretation. */
    {INT32_MIN,            1,  1},
    {INT32_MAX,            1,  1}
};

static struct {
    int8_t before;
    int8_t values[CAPACITY];
    int8_t after;
} input;

static int32_t reference[CAPACITY];

static struct {
    uint32_t before;
    int32_t values[CAPACITY];
    uint32_t after;
} output;

static void prepare_buffers(void)
{
    input.before = 0x35;
    input.after = 0x6a;

    output.before = UINT32_C(0x13579bdf);
    output.after = UINT32_C(0x2468ace0);

    for (unsigned int i = 0; i < CAPACITY; ++i) {
        input.values[i] = (int8_t)((int)i - 128);
        output.values[i] = UNWRITTEN;
    }
}

static void run_mode3(const TestCase *p, size_t count)
{
    const uint64_t cfg =
        UINT64_C(3) |
        ((uint64_t)p->shift << 3) |
        ((uint64_t)count << 9);

    __asm__ volatile(
        ".insn r 0x2b, 0, 0, x0, %0, x0\n\t"
        ".insn r 0x2b, 1, 0, x0, %1, %2\n\t"
        ".insn r 0x2b, 2, 0, x0, %3, x0\n\t"
        ".insn r 0x2b, 3, 0, x0, %4, x0\n\t"
        ".insn r 0x2b, 4, 0, x0, x0, x0"
        :
        : "r"(cfg),
          "r"((uintptr_t)input.values),
          "r"((uintptr_t)output.values),
          "r"(p->multiplier),
          "r"((intptr_t)p->input_zp)
        : "memory"
    );
}

static int check_output(
    unsigned int case_id,
    const TestCase *p,
    size_t count
)
{
    for (size_t i = 0; i < CAPACITY; ++i) {
        const int32_t wanted =
            i < count ? reference[i] : UNWRITTEN;

        if (output.values[i] != wanted) {
            printf(
                "FAIL case %u index=%u got=%d expected=%d\n",
                case_id, (unsigned int)i,
                (int)output.values[i], (int)wanted
            );

            printf(
                "Parameters: count=%u zp=%d multiplier=%lld shift=%u\n",
                (unsigned int)count, (int)p->input_zp,
                (long long)p->multiplier, p->shift
            );

            return 1;
        }

        if (input.values[i] != (int)i - 128) {
            printf(
                "FAIL case %u: input index %u changed\n",
                case_id, (unsigned int)i
            );
            return 1;
        }
    }

    if (input.before != 0x35 ||
        input.after != 0x6a ||
        output.before != UINT32_C(0x13579bdf) ||
        output.after != UINT32_C(0x2468ace0)) {
        printf("FAIL case %u: guard changed\n", case_id);
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

            prepare_buffers();

            requantize_i8_to_i32(
                input.values,
                reference,
                count,
                p->multiplier,
                p->shift,
                p->input_zp
            );

            run_mode3(p, count);

            if (check_output(completed, p, count)) {
                return 1;
            }

            ++completed;
            outputs += (unsigned int)count;
        }
    }

    printf(
        "PASS: %u cases, %u outputs match scalar reference; "
        "inputs, guards, and unused output slots unchanged\n",
        completed, outputs
    );

    return 0;
}