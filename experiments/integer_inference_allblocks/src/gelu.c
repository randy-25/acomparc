#include "gelu.h"

void gelu_multiply_scalar_i8(
    const int8_t *input,
    int32_t *output,
    size_t count,
    int32_t input_zero_point,
    int32_t centered_scalar
)
{
    for (size_t i = 0; i < count; ++i) {
        const int64_t centered =
            (int64_t)input[i] - input_zero_point;

        const int64_t product = centered * centered_scalar;

        output[i] = (int32_t)product;
    }
}

void gelu_cube_i8(
    const int8_t *input,
    int32_t *output,
    size_t count,
    int32_t input_zero_point
)
{
    for (size_t i = 0; i < count; ++i) {
        const int32_t centered =
            (int32_t)input[i] - input_zero_point;

        const int32_t square = centered * centered;

        output[i] = square * centered;
    }
}

void gelu_add_scaled_i8(
    const int8_t *linear,
    const int8_t *cubic,
    int64_t *output,
    size_t count,
    int32_t linear_zero_point,
    int32_t cubic_zero_point,
    int64_t linear_multiplier,
    int64_t cubic_multiplier
)
{
    for (size_t i = 0; i < count; ++i) {
        const int64_t centered_linear =
            (int64_t)linear[i] - linear_zero_point;

        const int64_t centered_cubic =
            (int64_t)cubic[i] - cubic_zero_point;

        const int64_t linear_product =
            centered_linear * linear_multiplier;

        const int64_t cubic_product =
            centered_cubic * cubic_multiplier;

        output[i] = linear_product + cubic_product;
    }
}

void gelu_tanh_lut_i8(
    const int8_t *input,
    int32_t *output,
    size_t count,
    const int32_t *table
)
{
    for (size_t i = 0; i < count; ++i) {
        const size_t index =
            (size_t)((int32_t)input[i] + 128);

        output[i] = table[index];
    }
}


void gelu_add_scalar_i32(
    const int32_t *input,
    int32_t *output,
    size_t count,
    int32_t scalar
)
{
    for (size_t i = 0; i < count; ++i) {
        output[i] = (int32_t)((int64_t)input[i] + scalar);
    }
}

void gelu_multiply_i8(
    const int8_t *left,
    const int8_t *right,
    int32_t *output,
    size_t count,
    int32_t left_zero_point,
    int32_t right_zero_point
)
{
    for (size_t i = 0; i < count; ++i) {
        const int32_t centered_left =
            (int32_t)left[i] - left_zero_point;

        const int32_t centered_right =
            (int32_t)right[i] - right_zero_point;

        output[i] = centered_left * centered_right;
    }
}