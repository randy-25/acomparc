#ifndef BLOCK0_MLP_OUT_RESIDUAL_GOLDEN_H
#define BLOCK0_MLP_OUT_RESIDUAL_GOLDEN_H

#include <stdint.h>

#define BLOCK0_MLP_OUT_RESIDUAL_CASE_COUNT 8u
#define BLOCK0_MLP_OUT_RESIDUAL_INPUT_ELEMENTS 16384u
#define BLOCK0_MLP_OUT_RESIDUAL_ELEMENTS 4096u

typedef struct {
    int8_t t104[16384];
    int32_t t73[4096];
    int32_t accumulator[4096];
    int8_t t105[4096];
    int32_t t106[4096];
    int32_t t107[4096];
} Block0MlpOutResidualGolden;

extern const int8_t block0_mlp_out_residual_t0[8][4096];
extern const Block0MlpOutResidualGolden
    block0_mlp_out_residual_local[8];
extern const Block0MlpOutResidualGolden
    block0_mlp_out_residual_chained[8];

#endif
