#ifndef BLOCK0_LN1_GOLDEN_H
#define BLOCK0_LN1_GOLDEN_H

#include <stdint.h>

/* Source SHA-256: 06481f6c5151a83e3c01896d314d902ee9608773d94834848777ccc7282d87a9
 * Selected integer candidate starting from encoded T0.
 * Feature tensors are flattened in token-major order.
 * Case 0: benchmark
 * Case 1: calibration_0
 * Case 2: calibration_1
 * Case 3: calibration_2
 * Case 4: calibration_3
 * Case 5: calibration_4
 * Case 6: calibration_5
 * Case 7: calibration_6
 */

extern const int64_t block0_ln1_t33_fine_golden[8][4096];
extern const int32_t block0_ln1_t33_golden[8][4096];
extern const int32_t block0_ln1_t34_golden[8][32];
extern const int32_t block0_ln1_t35_golden[8][32];
extern const int32_t block0_ln1_t36_golden[8][32];
extern const int64_t block0_ln1_t42_fine_golden[8][4096];
extern const int32_t block0_ln1_t42_golden[8][4096];
extern const int64_t block0_ln1_t44_golden[8][4096];
extern const int8_t block0_ln1_t45_golden[8][4096];

#endif
