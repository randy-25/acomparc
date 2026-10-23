#include <stdint.h>
#include <stdio.h>

#include "projection.h"

enum {
    M = 3,
    N = 3,
    K = 4
};

/* Same A[M,K] as test_xtile_run_i8.c. */
static const int8_t a[M][K] = {
    { 1,  2, 3,  4},
    {-1,  0, 1,  2},
    { 0, -2, 2, -1}
};

/* Same BT[N,K] as test_xtile_run_i8.c. */
static const int8_t bt[N][K] = {
    { 1, 0, -1, 2},
    {-2, 1,  0, 1},
    {-3, 1, -2, 0}
};

/* Input zero point = -1; weight zero point = 0. */
static const int32_t expected[M][N] = {
    { 8,  4, -11},
    { 4,  4,  -3},
    {-2, -3, -10}
};

static struct {
    uint32_t before;
    int32_t c[M][N];
    uint32_t after;
} output;

int main(void)
{
    const uint32_t before_sentinel = UINT32_C(0x13579bdf);
    const uint32_t after_sentinel = UINT32_C(0x2468ace0);

    output.before = before_sentinel;
    output.after = after_sentinel;

    for (unsigned int m = 0; m < M; ++m) {
        for (unsigned int n = 0; n < N; ++n) {
            output.c[m][n] = INT32_C(0x55555555);
        }
    }

    printf("Scalar projection INT8/INT8: M=%u N=%u K=%u\n",
           (unsigned int)M,
           (unsigned int)N,
           (unsigned int)K);

    /* Original scalar kernel; no custom instructions. */
    projection_accumulate_i8(
        &a[0][0],
        &bt[0][0],
        NULL,
        &output.c[0][0],
        M,                  /* token_count */
        K,                  /* input_features */
        N,                  /* output_features */
        -1                  /* input_zero_point */
    );

    unsigned int mismatches = 0;

    for (unsigned int m = 0; m < M; ++m) {
        for (unsigned int n = 0; n < N; ++n) {
            const int32_t actual = output.c[m][n];
            const int32_t wanted = expected[m][n];

            printf("C[%u][%u]: actual=%d expected=%d %s\n",
                   m, n,
                   (int)actual,
                   (int)wanted,
                   actual == wanted ? "PASS" : "FAIL");

            if (actual != wanted) {
                ++mismatches;
            }
        }
    }

    if (output.before != before_sentinel ||
        output.after != after_sentinel) {
        printf("FAIL: output sentinel changed\n");
        return 1;
    }

    if (mismatches != 0) {
        printf("FAIL: %u output mismatches\n", mismatches);
        return 1;
    }

    printf("PASS: 9 INT32 outputs; output sentinels unchanged\n");
    return 0;
}