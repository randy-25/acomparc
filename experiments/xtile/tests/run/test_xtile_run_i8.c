#include <stdint.h>
#include <stdio.h>

enum {
    M = 3,
    N = 3,
    K = 4,
    TILE_ROWS = 2,
    TILE_COLUMNS = 2
};

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

/* A[M,K], signed INT8. */
static const int8_t a[M][K] = {
    { 1,  2, 3,  4},
    {-1,  0, 1,  2},
    { 0, -2, 2, -1}
};

/* BT[N,K], signed INT8: each row is one output column's weights. */
static const int8_t bt[N][K] = {
    { 1, 0, -1, 2},
    {-2, 1,  0, 1},
    {-3, 1, -2, 0}
};

/*
 * A zero point = -1; B zero point = 0.
 *
 * Centered A:
 *   {2,  3, 4, 5}
 *   {0,  1, 2, 3}
 *   {1, -1, 3, 0}
 *
 * Expected C = centered_A * BT^T.
 */
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

    /*
     * A_TYPE = 0, B_TYPE = 0: INT8/INT8.
     * Tile dimensions are encoded minus one.
     */
    const uint64_t cfg =
        ((uint64_t)(TILE_ROWS - 1) << 2) |
        ((uint64_t)(TILE_COLUMNS - 1) << 5) |
        ((uint64_t)M << 8) |
        ((uint64_t)N << 18) |
        ((uint64_t)K << 28);

    output.before = before_sentinel;
    output.after = after_sentinel;

    for (unsigned int m = 0; m < M; ++m) {
        for (unsigned int n = 0; n < N; ++n) {
            output.c[m][n] = INT32_C(0x55555555);
        }
    }

    printf("xtile.run INT8/INT8: M=%u N=%u K=%u tile=%ux%u\n",
           (unsigned int)M,
           (unsigned int)N,
           (unsigned int)K,
           (unsigned int)TILE_ROWS,
           (unsigned int)TILE_COLUMNS);

    xtile_cfg(cfg);
    xtile_set_ab(a, bt);
    xtile_set_c(output.c);
    xtile_set_zp(-1, 0);

    xtile_run();

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