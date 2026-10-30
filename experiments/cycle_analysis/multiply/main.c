#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

extern int32_t multiply_i8(int8_t a, int8_t b, int32_t zero_point);

static inline uint64_t read_cycles(void)
{
    uint64_t value;
    __asm__ volatile ("csrr %0, mcycle" : "=r"(value) : : "memory");
    return value;
}

int main(void)
{
    volatile int8_t input_a = 12;
    volatile int8_t input_b = -7;
    volatile int32_t input_zero_point = 2;

    int8_t a = input_a;
    int8_t b = input_b;
    int32_t zero_point = input_zero_point;

    // // Measure the first call
    // uint64_t start = read_cycles();
    // int32_t result = multiply_i8(a, b, zero_point);
    // uint64_t end = read_cycles();

    // Execute once before the measured call.
    int32_t warmup = multiply_i8(a, b, zero_point);
    if (warmup != -70)
        return 1;

    uint64_t start = read_cycles();
    int32_t result = multiply_i8(a, b, zero_point);
    uint64_t end = read_cycles();

    printf("Result: %" PRId32 " (expected -70)\n", result);
    printf("Raw measured cycles: %" PRIu64 "\n", end - start);

    return result == -70 ? 0 : 1;
}
