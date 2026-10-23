#include <stdint.h>
#include <stdio.h>

enum {
    OUTPUT_CAPACITY = 90,
    MAX_K = 1023
};

/*
 * Mode bits match cfg[1:0]:
 *   0: INT8  / INT8
 *   1: INT32 / INT8
 *   2: INT8  / INT32
 *   3: INT32 / INT32
 */
typedef struct {
    const char *name;
    const void *a;
    const void *bt;
    const int64_t *expected;
    unsigned int mode;
    unsigned int m;
    unsigned int n;
    unsigned int k;
    unsigned int tile_rows;
    unsigned int tile_columns;
    int32_t a_zp;
    int32_t b_zp;
} TestCase;

static inline void xtile_cfg(uint64_t cfg)
{
    __asm__ volatile(
        ".insn r 0x0b, 0, 0, x0, %0, x0"
        :
        : "r"(cfg)
    );
}

static inline void xtile_set_ab(const void *a, const void *bt)
{
    __asm__ volatile(
        ".insn r 0x0b, 1, 0, x0, %0, %1"
        :
        : "r"((uintptr_t)a), "r"((uintptr_t)bt)
    );
}

static inline void xtile_set_c(void *c)
{
    __asm__ volatile(
        ".insn r 0x0b, 2, 0, x0, %0, x0"
        :
        : "r"((uintptr_t)c)
    );
}

static inline void xtile_set_zp(int32_t a_zp, int32_t b_zp)
{
    __asm__ volatile(
        ".insn r 0x0b, 3, 0, x0, %0, %1"
        :
        : "r"((intptr_t)a_zp), "r"((intptr_t)b_zp)
    );
}

static inline void xtile_run(void)
{
    __asm__ volatile(
        ".insn r 0x0b, 4, 0, x0, x0, x0"
        :
        :
        : "memory"
    );
}

/* Separate, naturally aligned buffers for each output width. */
static struct {
    uint64_t before;
    int32_t values[OUTPUT_CAPACITY];
    uint64_t after;
} output32;

static struct {
    uint64_t before;
    int64_t values[OUTPUT_CAPACITY];
    uint64_t after;
} output64;

static unsigned int passed_cases;
static unsigned int checked_outputs;

static int run_case(const TestCase *test)
{
    const uint64_t guard_before = UINT64_C(0x0123456789abcdef);
    const uint64_t guard_after = UINT64_C(0xfedcba9876543210);
    const int32_t unused32 = INT32_C(0x55555555);
    const int64_t unused64 = INT64_C(0x5555555555555555);

    const unsigned int count = test->m * test->n;
    const int wide = test->mode != 0;

    if (count > OUTPUT_CAPACITY) {
        printf("FAIL: test output exceeds buffer capacity\n");
        return 1;
    }

    output32.before = guard_before;
    output32.after = guard_after;
    output64.before = guard_before;
    output64.after = guard_after;

    for (unsigned int i = 0; i < OUTPUT_CAPACITY; ++i) {
        output32.values[i] = unused32;
        output64.values[i] = unused64;
    }

    const uint64_t cfg =
        (uint64_t)test->mode |
        ((uint64_t)(test->tile_rows - 1u) << 2) |
        ((uint64_t)(test->tile_columns - 1u) << 5) |
        ((uint64_t)test->m << 8) |
        ((uint64_t)test->n << 18) |
        ((uint64_t)test->k << 28);

    xtile_cfg(cfg);
    xtile_set_ab(test->a, test->bt);
    xtile_set_c(
        wide ? (void *)output64.values : (void *)output32.values
    );
    xtile_set_zp(test->a_zp, test->b_zp);
    xtile_run();

    for (unsigned int i = 0; i < count; ++i) {
        const int64_t actual =
            wide ? output64.values[i] : (int64_t)output32.values[i];

        if (actual != test->expected[i]) {
            printf("Case %u FAIL: %s, element %u\n",
                   passed_cases, test->name, i);

            printf("actual=%lld expected=%lld\n",
                   (long long)actual,
                   (long long)test->expected[i]);

            return 1;
        }
    }

    /* Detect writes beyond this case's actual output length. */
    for (unsigned int i = count; i < OUTPUT_CAPACITY; ++i) {
        if ((wide && output64.values[i] != unused64) ||
            (!wide && output32.values[i] != unused32)) {
            printf("Case %u FAIL: output overrun at element %u\n",
                   passed_cases, i);
            return 1;
        }
    }

    if (output32.before != guard_before ||
        output32.after != guard_after ||
        output64.before != guard_before ||
        output64.after != guard_after) {
        printf("Case %u FAIL: output sentinel changed\n",
               passed_cases);
        return 1;
    }

    printf("Case %u: %s PASS "
           "(%u outputs, tile=%ux%u)\n",
           passed_cases, test->name, count,
           test->tile_rows, test->tile_columns);

    ++passed_cases;
    checked_outputs += count;
    return 0;
}

/* Small scalar operands. */
static const int8_t i8_maximum = INT8_MAX;
static const int8_t i8_minimum = INT8_MIN;
static const int32_t i32_maximum = INT32_MAX;
static const int32_t i32_minimum = INT32_MIN;

/* Mixed-width dot products. */
static const int32_t mixed_a32[2] = {
    2000000000, -2000000000
};

static const int8_t mixed_bt8[3][2] = {
    { 127,  127},
    { 127, -128},
    {-128,  127}
};

static const int8_t mixed_a8[2] = {
    127, -128
};

static const int32_t mixed_bt32[2][2] = {
    { 2000000000, -2000000000},
    {-2000000000,  2000000000}
};

/* Both matrices use these two values, with K=1. */
static const int32_t limits32[2] = {
    INT32_MIN, INT32_MAX
};

/*
 * Each product is:
 *   INT32_MIN * (INT32_MAX - (-1)) = -2^62.
 * Two products sum exactly to INT64_MIN.
 */
static const int32_t minimum_a[2] = {
    INT32_MIN, INT32_MIN
};

static const int32_t minimum_bt[2] = {
    INT32_MAX, INT32_MAX
};

/*
 * A zero point = -1; B zero point = 0.
 * All partial sums fit INT64.
 * Final sum is exactly INT64_MAX.
 */
static const int32_t maximum_a[4] = {
    INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX - 1
};

static const int32_t maximum_bt[4] = {
    INT32_MAX, INT32_MAX, 1, 1
};

static const int64_t expected_i8_positive[1] = {
    INT64_C(65025)
};

static const int64_t expected_i8_negative[1] = {
    -INT64_C(65025)
};

static const int64_t expected_mixed32_8[3] = {
    0, INT64_C(510000000000), -INT64_C(510000000000)
};

static const int64_t expected_mixed8_32[2] = {
    INT64_C(510000000000), -INT64_C(510000000000)
};

static const int64_t expected_center_positive[1] = {
    INT64_C(1095216660225)
};

static const int64_t expected_center_negative[1] = {
    -INT64_C(1095216660225)
};

static const int64_t expected_limits32[4] = {
     INT64_C(4611686018427387904),
    -INT64_C(4611686016279904256),
    -INT64_C(4611686016279904256),
     INT64_C(4611686014132420609)
};

static const int64_t expected_minimum[1] = {
    INT64_MIN
};

static const int64_t expected_maximum[1] = {
    INT64_MAX
};

static int8_t long_positive[MAX_K];
static int8_t long_negative[MAX_K];

static const int64_t expected_long_positive[1] = {
    INT64_C(66520575)
};

static const int64_t expected_long_negative[1] = {
    -INT64_C(66520575)
};

/* A[9,3], BT[10,3], C[9,10]. */
static int8_t tile_a[9 * 3];
static int8_t tile_bt[10 * 3];
static int64_t tile_expected[9 * 10];

int main(void)
{
    for (unsigned int i = 0; i < MAX_K; ++i) {
        long_positive[i] = INT8_MAX;
        long_negative[i] = INT8_MIN;
    }

    /*
     * Entry order:
     * name, A, BT, expected,
     * mode, M, N, K, tile_rows, tile_columns, A_zp, B_zp.
     */
    static const TestCase cases[] = {
        {
            "INT8 positive extremes",
            &i8_maximum, &i8_maximum, expected_i8_positive,
            0, 1, 1, 1, 8, 8, -128, -128
        },
        {
            "INT8 negative centered operand",
            &i8_minimum, &i8_maximum, expected_i8_negative,
            0, 1, 1, 1, 1, 1, 127, -128
        },
        {
            "INT32 x INT8 beyond INT32 output range",
            mixed_a32, mixed_bt8, expected_mixed32_8,
            1, 1, 3, 2, 2, 2, 0, 0
        },
        {
            "INT8 x INT32 beyond INT32 output range",
            mixed_a8, mixed_bt32, expected_mixed8_32,
            2, 1, 2, 2, 1, 1, 0, 0
        },
        {
            "A centering requires 33 bits",
            &i32_maximum, &i8_maximum, expected_center_positive,
            1, 1, 1, 1, 8, 8, INT32_MIN, -128
        },
        {
            "B centering requires 33 bits",
            &i8_maximum, &i32_minimum, expected_center_negative,
            2, 1, 1, 1, 8, 8, -128, INT32_MAX
        },
        {
            "INT32 x INT32 signed products",
            limits32, limits32, expected_limits32,
            3, 2, 2, 1, 2, 2, 0, 0
        },
        {
            "exact INT64_MIN",
            minimum_a, minimum_bt, expected_minimum,
            3, 1, 1, 2, 1, 1, 0, -1
        },
        {
            "exact INT64_MAX",
            maximum_a, maximum_bt, expected_maximum,
            3, 1, 1, 4, 1, 1, -1, 0
        },
        {
            "K=1023 positive INT8 accumulation",
            long_positive, long_positive, expected_long_positive,
            0, 1, 1, MAX_K, 8, 8, -128, -128
        },
        {
            "K=1023 negative INT8 accumulation",
            long_negative, long_positive, expected_long_negative,
            0, 1, 1, MAX_K, 8, 8, 127, -128
        }
    };

    printf("xtile.run arithmetic and tile edge tests\n");

    for (unsigned int i = 0;
         i < (unsigned int)(sizeof(cases) / sizeof(cases[0]));
         ++i) {
        if (run_case(&cases[i])) {
            return 1;
        }
    }

    /*
     * A row m = [m, m+1, m+2].
     * BT row n = [n, 1, -1].
     *
     * C[m,n] = m*n + (m+1) - (m+2) = m*n - 1.
     */
    for (unsigned int m = 0; m < 9; ++m) {
        tile_a[m * 3] = (int8_t)m;
        tile_a[m * 3 + 1] = (int8_t)(m + 1);
        tile_a[m * 3 + 2] = (int8_t)(m + 2);
    }

    for (unsigned int n = 0; n < 10; ++n) {
        tile_bt[n * 3] = (int8_t)n;
        tile_bt[n * 3 + 1] = 1;
        tile_bt[n * 3 + 2] = -1;
    }

    for (unsigned int m = 0; m < 9; ++m) {
        for (unsigned int n = 0; n < 10; ++n) {
            tile_expected[m * 10 + n] =
                (int64_t)m * (int64_t)n - 1;
        }
    }

    static const unsigned int shapes[][2] = {
        {1, 1},
        {1, 8},
        {8, 1},
        {4, 8},
        {8, 4},
        {8, 8}
    };

    for (unsigned int i = 0;
         i < (unsigned int)(sizeof(shapes) / sizeof(shapes[0]));
         ++i) {

        const TestCase test = {
            "9x10 output with tile boundaries",
            tile_a, tile_bt, tile_expected,
            0, 9, 10, 3, shapes[i][0], shapes[i][1], 0, 0
        };

        if (run_case(&test)) {
            return 1;
        }
    }

    printf("PASS: %u cases, %u outputs, "
           "sentinels and unused output slots unchanged\n",
           passed_cases, checked_outputs);

    return 0;
}