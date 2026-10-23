#ifndef BLOCK0_ATTENTION_SCORES_GOLDEN_H
#define BLOCK0_ATTENTION_SCORES_GOLDEN_H

#include <stdint.h>

#define BLOCK0_ATTN_SCORE_CASE_COUNT 8
#define BLOCK0_ATTN_SCORE_ELEMENTS 16384

/* Source SHA-256: 06481f6c5151a83e3c01896d314d902ee9608773d94834848777ccc7282d87a9
 * T59: INT32 [16,32,32], zero point 0.
 * Scale: 0.0004575034413775367.
 * Flattened index: (head * 32 + query_token) * 32 + key_token.
 * Includes all scores before masking.
 * local: Q/K from reference T45.
 * chained: Q/K from selected integer LN1 T45.
 * Case 0: benchmark
 * Case 1: calibration_0
 * Case 2: calibration_1
 * Case 3: calibration_2
 * Case 4: calibration_3
 * Case 5: calibration_4
 * Case 6: calibration_5
 * Case 7: calibration_6
 */

extern const int32_t block0_attention_scores_local_golden[8][16384];
extern const int32_t block0_attention_scores_chained_golden[8][16384];

#endif
