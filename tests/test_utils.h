#pragma once

#include <random>

constexpr int kSIZE = 1 << 24;
constexpr int kNPOT = kSIZE - 3;  // non-power-of-two

template<std::integral T, std::size_t N>
inline void fill_array_random(std::array<T, N>& a, int max_val)
{
    std::random_device rd;
    std::mt19937 gen(rd());  // initialize Mersenne Twister engine
    std::uniform_int_distribution<T> distrib(0, max_val);

    for (int i = 0; i < a.size(); ++i)
    {
        a[i] = distrib(gen);
    }
}
