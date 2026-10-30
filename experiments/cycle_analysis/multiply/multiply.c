#include <stdint.h>

int32_t multiply_i8(int8_t a, int8_t b, int32_t zero_point)
{
    return ((int32_t)a - zero_point) * (int32_t)b;
}
