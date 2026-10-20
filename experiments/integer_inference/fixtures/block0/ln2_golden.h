#ifndef BLOCK0_LN2_GOLDEN_H
#define BLOCK0_LN2_GOLDEN_H

#include <stdint.h>

#define BLOCK0_LN2_GOLDEN_CASE_COUNT 8u
#define BLOCK0_LN2_GOLDEN_TOKEN_COUNT 32u
#define BLOCK0_LN2_GOLDEN_ELEMENTS 4096u

typedef struct {
    int32_t t73[4096];
    int32_t t74[32];
    int64_t t76_fine[4096];
    int32_t t76[4096];
    int32_t t77[32];
    int32_t t78[32];
    int32_t t79[32];
    int64_t t85_fine[4096];
    int32_t t85[4096];
    int64_t t87[4096];
    int8_t t88[4096];
} Block0Ln2Golden;

extern const int8_t block0_ln2_t0[8][4096];
extern const Block0Ln2Golden block0_ln2_local[8];
extern const Block0Ln2Golden block0_ln2_chained[8];

#endif
