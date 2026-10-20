#ifndef BLOCK0_OUT_RESIDUAL_GOLDEN_H
#define BLOCK0_OUT_RESIDUAL_GOLDEN_H

#include <stdint.h>

#define BLOCK0_OUT_RESIDUAL_CASE_COUNT 8u
#define BLOCK0_OUT_RESIDUAL_ELEMENTS 4096u

typedef struct {
    int8_t t70[BLOCK0_OUT_RESIDUAL_ELEMENTS];
    int32_t accumulator[BLOCK0_OUT_RESIDUAL_ELEMENTS];
    int8_t t71[BLOCK0_OUT_RESIDUAL_ELEMENTS];
    int32_t t72[BLOCK0_OUT_RESIDUAL_ELEMENTS];
    int32_t t73[BLOCK0_OUT_RESIDUAL_ELEMENTS];
} Block0OutResidualGolden;

extern const int8_t block0_out_residual_t0[8][4096];
extern const Block0OutResidualGolden block0_out_residual_local[8];
extern const Block0OutResidualGolden block0_out_residual_chained[8];

#endif
