#ifndef BLOCK0_LN1_SUM_DATA_H
#define BLOCK0_LN1_SUM_DATA_H

#include <stdint.h>

/*
 * Source block SHA-256: 06481f6c5151a83e3c01896d314d902ee9608773d94834848777ccc7282d87a9
 * T0: INT8, scale 2^-8, zero point 0.
 * T31: INT32 feature sums, scale 2^-15, zero point 0.
 * Input indexing: token * 128 + feature.
 * Case 0: benchmark
 * Case 1: calibration_0
 * Case 2: calibration_1
 * Case 3: calibration_2
 * Case 4: calibration_3
 * Case 5: calibration_4
 * Case 6: calibration_5
 * Case 7: calibration_6
 */

#define BLOCK0_LN1_CASE_COUNT 8
#define BLOCK0_LN1_TOKEN_COUNT 32
#define BLOCK0_LN1_FEATURE_COUNT 128

extern const int8_t block0_ln1_t0[
    BLOCK0_LN1_CASE_COUNT
][BLOCK0_LN1_TOKEN_COUNT * BLOCK0_LN1_FEATURE_COUNT];

extern const int32_t block0_ln1_t31_golden[
    BLOCK0_LN1_CASE_COUNT
][BLOCK0_LN1_TOKEN_COUNT];

#endif
