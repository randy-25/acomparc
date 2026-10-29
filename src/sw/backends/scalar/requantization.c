#include "requantization.h"

/* Integer-only scalar counterparts of the five existing XREQUANT modes.
 * Preconditions in requantization.h apply, including INT64 product/rounding
 * bounds and arithmetic signed right shift on the GCC targets used here.
 */
static inline int active(size_t count, unsigned int shift, unsigned int minimum, unsigned int maximum)
{
    if (count == 0) return 0;
    if (count > UINT32_MAX || shift < minimum || shift > maximum) __builtin_trap();
    return 1;
}

static inline int8_t clamp_i8(int64_t value)
{
    if (value < -128) return -128;
    if (value > 127) return 127;
    return (int8_t)value;
}

void requantize_shift_i64_to_i32(const int64_t *input, int32_t *output,
                                size_t count, unsigned int shift)
{
    if (!active(count, shift, 1, 63)) return;
    const int64_t rounding = INT64_C(1) << (shift-1);
    for (size_t i = 0; i < count; ++i) output[i] = (int32_t)((input[i]+rounding) >> shift);
}

void requantize_i64_to_i8(const int64_t *input, int8_t *output, size_t count,
                         int64_t multiplier, unsigned int shift, int32_t zero_point)
{
    if (!active(count, shift, 1, 63)) return;
    const int64_t rounding = INT64_C(1) << (shift-1);
    for (size_t i = 0; i < count; ++i)
        output[i] = clamp_i8(((input[i]*multiplier+rounding) >> shift)+zero_point);
}

void requantize_i32_to_i8(const int32_t *input, int8_t *output, size_t count,
                         int64_t multiplier, unsigned int shift, int32_t zero_point)
{
    if (!active(count, shift, 1, 63)) return;
    const int64_t rounding = INT64_C(1) << (shift-1);
    for (size_t i = 0; i < count; ++i)
        output[i] = clamp_i8((((int64_t)input[i]*multiplier+rounding) >> shift)+zero_point);
}

void requantize_i8_to_i32(const int8_t *input, int32_t *output, size_t count,
                         int64_t multiplier, unsigned int shift, int32_t zero_point)
{
    if (!active(count, shift, 1, 63)) return;
    const int64_t rounding = INT64_C(1) << (shift-1);
    for (size_t i = 0; i < count; ++i) {
        const int64_t centered = (int64_t)input[i]-zero_point;
        output[i] = (int32_t)((centered*multiplier+rounding) >> shift);
    }
}

void requantize_two_round_i32_to_i8(const int32_t *input, int8_t *output, size_t count,
                                  int32_t q31_multiplier, unsigned int shift, int32_t zero_point)
{
    if (!active(count, shift, 0, 31)) return;
    if (q31_multiplier <= 0) __builtin_trap();
    const uint64_t mask = (UINT64_C(1) << shift)-1;
    for (size_t i = 0; i < count; ++i) {
        const int64_t product = (int64_t)input[i]*q31_multiplier;
        const int64_t high = (product+(INT64_C(1) << 30)) >> 31;
        const uint64_t remainder = (uint64_t)high & mask;
        const uint64_t threshold = (mask >> 1)+(high < 0);
        const int64_t rounded = (high >> shift)+(remainder > threshold);
        output[i] = clamp_i8(rounded+zero_point);
    }
}
