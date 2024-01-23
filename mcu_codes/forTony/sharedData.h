#pragma once
#include <cstdint>
#include <cstddef>

//define return struct
struct RetResult
{
    uint32_t inputNumber = 0;
    size_t max_idx = 0;
    int8_t max_value = -128;
};