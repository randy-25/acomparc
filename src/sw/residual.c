#include "residual.h"

void residual_add_i8_i32(
    const int8_t *residual,
    const int32_t *branch,
    int32_t *output,
    size_t count,
    int64_t alignment,
    int32_t residual_zero_point
)
{
    for (size_t i = 0; i < count; ++i) {
        const int64_t centered =
            (int64_t)residual[i] - residual_zero_point;

        const int64_t aligned = centered * alignment;

        output[i] = (int32_t)((int64_t)branch[i] + aligned);
    }
}

void residual_add_i32(
    const int32_t *left,
    const int32_t *right,
    int32_t *output,
    size_t count
)
{
    for (size_t i = 0; i < count; ++i) {
        const int64_t sum = (int64_t)left[i] + right[i];

        output[i] = (int32_t)sum;
    }
}