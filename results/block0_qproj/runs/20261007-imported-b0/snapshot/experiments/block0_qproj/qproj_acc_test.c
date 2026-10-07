
#include <stdint.h>
#include <stdio.h>
#include "qproj_data.h"
#include "qproj_y_ref.h"

static int8_t output_q[32 * 128];


#define OUTPUT_ZERO_POINT (-17)
#define OUTPUT_SCALE_Q INT64_C(1205592262)
#define OUTPUT_SCALE_SHIFT 39

static int8_t requantize_qproj(int32_t acc)
{
    int64_t product = (int64_t)acc * OUTPUT_SCALE_Q;

    int64_t result =
        (product + (INT64_C(1) << (OUTPUT_SCALE_SHIFT - 1)))
        >> OUTPUT_SCALE_SHIFT;

    result += OUTPUT_ZERO_POINT;

    if (result > 127)  result = 127;
    if (result < -128) result = -128;

    return (int8_t)result;
}

__attribute__((noinline))
void qproj_accumulate(void)
{
    for (int t = 0; t < 32; ++t) {
        for (int o = 0; o < 128; ++o) {
            int32_t acc = 0;

            for (int k = 0; k < 128; ++k) {
                int32_t x =
                    (int32_t)input[t * 128 + k]
                    - INPUT_ZERO_POINT;

                int32_t w =
                    (int32_t)weights[o * 128 + k];

                acc += x * w;
            }
            // acc now contains the complete sum of 128 products.
            output_q[t * 128 + o] = requantize_qproj(acc);
        }
    }
    
}

#ifdef __riscv
    void setStats(int enable);
#endif
int main(void)
{
    #ifdef __riscv
        setStats(1);       // Record starting counters.
    #endif

        qproj_accumulate();

    #ifdef __riscv
        setStats(0);       // Calculate counter differences.
    #endif

    int errors = 0;
    for (int i = 0; i < 32 * 128; ++i) {
        if (output_q[i] != expected_y[i])
            ++errors;
    }

    printf("Checked 4096 INT8 outputs; mismatches: %d\n", errors);
    printf("%s\n", errors == 0 ? "PASS" : "FAIL");

    return errors == 0 ? 0 : 1;
}