#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "qproj_data.h"

static int32_t output_acc[32 * 128];

static inline uint64_t read_cycles(void)
{
    uint64_t value;
    __asm__ volatile ("csrr %0, mcycle" : "=r"(value) : : "memory");
    return value;
}

__attribute__((noinline))
void qproj_accumulate_only(void)
{
    for (int t = 0; t < 32; ++t) {
        for (int o = 0; o < 128; ++o) {
            int32_t acc = 0;

            for (int k = 0; k < 128; ++k) {
                int32_t x =
                    (int32_t)input[t * 128 + k] - INPUT_ZERO_POINT;
                int32_t w =
                    (int32_t)weights[o * 128 + k];

                acc += x * w;
            }

            output_acc[t * 128 + o] = acc;
        }
    }
}

int main(void)
{
    uint64_t start = read_cycles();
    qproj_accumulate_only();
    uint64_t end = read_cycles();

    unsigned errors = 0;
    for (unsigned i = 0; i < 32u * 128u; ++i) {
        if (output_acc[i] != expected_acc[i])
            ++errors;
    }

    printf("MAC operations: 524288\n");
    printf("Raw accumulation cycles: %" PRIu64 "\n", end - start);
    printf("Checked 4096 INT32 outputs; mismatches: %u\n", errors);
    printf("%s\n", errors == 0 ? "PASS" : "FAIL");

    return errors == 0 ? 0 : 1;
}
