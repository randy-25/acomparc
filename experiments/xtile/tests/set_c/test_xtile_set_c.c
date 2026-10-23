#include <stdint.h>
#include <stdio.h>

static inline void xtile_cfg(uint64_t configuration)
{
    __asm__ volatile (
        ".insn r 0x0b, 0, 0, x0, %0, x0"
        :
        : "r" (configuration)
    );
}

static inline void xtile_set_c(int32_t *output)
{
    __asm__ volatile (
        ".insn r 0x0b, 2, 0, x0, %0, x0"
        :
        : "r" ((uintptr_t)output)
        : "memory"
    );
}

/* Flat storage for two INT32 matrices of shape [2,2]. */
static int32_t output_first[4] = {101, -202, 303, -404};
static int32_t output_second[4] = {505, -606, 707, -808};

static const int32_t expected_first[4] = {101, -202, 303, -404};
static const int32_t expected_second[4] = {505, -606, 707, -808};

static int check_buffers(unsigned int case_number)
{
    for (unsigned int i = 0; i < 4; ++i) {
        if (output_first[i] != expected_first[i] ||
            output_second[i] != expected_second[i]) {
            printf("Case %u: unexpected buffer modification at index %u\n",
                   case_number, i);
            return 1;
        }
    }

    return 0;
}

int main(void)
{
    /* INT8 operands, active tile 2x2, M=2, N=2, K=8. */
    const uint64_t configuration =
        (UINT64_C(1) << 2)  |
        (UINT64_C(1) << 5)  |
        (UINT64_C(2) << 8)  |
        (UINT64_C(2) << 18) |
        (UINT64_C(8) << 28);

    printf("xtile.set_c smoke test: %u cases\n", 3u);

    xtile_set_c(output_first);
    if (check_buffers(0u)) {
        return 1;
    }
    printf("Case %u: set_c before cfg PASS\n", 0u);

    xtile_cfg(configuration);
    xtile_set_c(output_first);
    if (check_buffers(1u)) {
        return 1;
    }
    printf("Case %u: set_c after cfg PASS\n", 1u);

    xtile_set_c(output_second);
    if (check_buffers(2u)) {
        return 1;
    }
    printf("Case %u: replacement address accepted PASS\n", 2u);

    printf("PASS: %u set_c smoke cases; both buffers unchanged\n", 3u);
    return 0;
}