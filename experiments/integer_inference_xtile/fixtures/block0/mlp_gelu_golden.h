#ifndef BLOCK0_MLP_GELU_GOLDEN_H
#define BLOCK0_MLP_GELU_GOLDEN_H

#include <stdint.h>

#define BLOCK0_MLP_GELU_CASE_COUNT 8u
#define BLOCK0_MLP_GELU_INPUT_ELEMENTS 4096u
#define BLOCK0_MLP_GELU_ELEMENTS 16384u

typedef struct {
    int8_t t88[4096];
    int32_t accumulator[16384];
    int8_t t89[16384];
    int8_t t92[16384];
    int32_t t93[16384];
    int8_t t94[16384];
    int8_t t96[16384];
    int8_t t97[16384];
    int8_t t99[16384];
    int32_t t101[16384];
    int32_t t102[16384];
    int8_t t103[16384];
    int8_t t104[16384];
} Block0MlpGeluGolden;

extern const int8_t block0_mlp_gelu_t0[8][4096];
extern const Block0MlpGeluGolden block0_mlp_gelu_local[8];
extern const Block0MlpGeluGolden block0_mlp_gelu_chained[8];

#endif
