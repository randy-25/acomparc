#ifndef BLOCK0_ATTENTION_GOLDEN_H
#define BLOCK0_ATTENTION_GOLDEN_H

#include <stdint.h>

#define BLOCK0_ATTENTION_CASE_COUNT 8u
#define BLOCK0_ATTENTION_SCORE_ELEMENTS 16384u
#define BLOCK0_ATTENTION_FEATURE_ELEMENTS 4096u

extern const int8_t block0_attention_t0[8][4096];
extern const int32_t block0_attention_shifted_local_golden[8][16384];
extern const int32_t block0_attention_exp_local_golden[8][16384];
extern const int32_t block0_attention_probabilities_local_golden[8][16384];
extern const int32_t block0_attention_context_local_golden[8][4096];
extern const int8_t block0_attention_output_local_golden[8][4096];
extern const int32_t block0_attention_shifted_chained_golden[8][16384];
extern const int32_t block0_attention_exp_chained_golden[8][16384];
extern const int32_t block0_attention_probabilities_chained_golden[8][16384];
extern const int32_t block0_attention_context_chained_golden[8][4096];
extern const int8_t block0_attention_output_chained_golden[8][4096];

#endif
