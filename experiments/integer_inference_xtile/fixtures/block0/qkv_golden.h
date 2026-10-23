#ifndef BLOCK0_QKV_GOLDEN_H
#define BLOCK0_QKV_GOLDEN_H

#include <stdint.h>

#define BLOCK0_QKV_CASE_COUNT 8
#define BLOCK0_QKV_ELEMENTS 4096

/* Source SHA-256: 06481f6c5151a83e3c01896d314d902ee9608773d94834848777ccc7282d87a9
 * local: reference T45 input.
 * chained: selected integer LN1 T45 input.
 * Feature arrays are flattened in token-major order.
 * Case 0: benchmark
 * Case 1: calibration_0
 * Case 2: calibration_1
 * Case 3: calibration_2
 * Case 4: calibration_3
 * Case 5: calibration_4
 * Case 6: calibration_5
 * Case 7: calibration_6
 */

extern const int8_t block0_qkv_input_local[8][4096];
extern const int8_t block0_qkv_input_chained[8][4096];
extern const int32_t block0_q_acc_local_golden[8][4096];
extern const int8_t block0_q_output_local_golden[8][4096];
extern const int32_t block0_q_acc_chained_golden[8][4096];
extern const int8_t block0_q_output_chained_golden[8][4096];
extern const int32_t block0_k_acc_local_golden[8][4096];
extern const int8_t block0_k_output_local_golden[8][4096];
extern const int32_t block0_k_acc_chained_golden[8][4096];
extern const int8_t block0_k_output_chained_golden[8][4096];
extern const int32_t block0_v_acc_local_golden[8][4096];
extern const int8_t block0_v_output_local_golden[8][4096];
extern const int32_t block0_v_acc_chained_golden[8][4096];
extern const int8_t block0_v_output_chained_golden[8][4096];

#endif
