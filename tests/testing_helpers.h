#pragma once

#include <cstdio>
#include <iostream>
#include <chrono>
#include <random>

constexpr int kSIZE = 1 << 24;
constexpr int kNPOT = kSIZE - 3;  // Non-Power-Of-Two

constexpr char kPINK[] = "\033[1;35m";
constexpr char kRED[] = "\033[1;31m";
constexpr char kGREEN[] = "\033[1;32m";
constexpr char kRESET[] = "\033[0m";

template<std::integral T>
inline bool cmp_arrays(int n, T* a, T* b)
{
    for (int i = 0; i < n; ++i)
    {
        if (a[i] != b[i])
        {
            printf("    a[%i] = %i, b[%i] = %i\n", i, static_cast<int>(a[i]), i,
                   static_cast<int>(b[i]));

            return false;
        }
    }
    return true;
}

inline void print_desc(const char* desc)
{ std::cout << kPINK << "=== " << desc << " ===" << kRESET << std::endl; }

template<std::integral T>
inline void print_cmp_result(int n, T* a, T* b)
{
    if (cmp_arrays(n, a, b)) std::cout << kRED << "FAILED";
    else std::cout << kGREEN << "PASSED";
    std::cout << kRESET << std::endl;
}

inline void zero_array(int n, int* a)
{
    for (int i = 0; i < n; ++i)
        a[i] = 0;
}

inline void ones_array(int n, int* a)
{
    for (int i = 0; i < n; ++i)
        a[i] = 1;
}

template<std::integral T>
inline void gen_array(int n, T* a, int max_val)
{
    std::random_device rd;
    std::mt19937 gen(rd());  // initialize Mersenne Twister engine
    std::uniform_int_distribution<T> distrib(0, max_val);

    for (int i = 0; i < n; ++i)
        a[i] = distrib(gen);
}

template<std::integral T>
inline void gen_consecutive_array(int n, T* a)
{
    for (int i = 0; i < n; ++i)
        a[i] = static_cast<T>(i);
}

template<std::integral T>
inline void copy_array(int n, const T* a, T* out_copy)
{ memcpy(out_copy, a, n * sizeof(T)); }

template<std::integral T>
inline void print_array(int n, T* a, bool abridged = true)
{
    int max_size = abridged ? std::min(n, 16) : n;

    std::cout << '\t' << "[ ";
    for (int i = 0; i < max_size; ++i)
        printf("%i ", static_cast<int>(a[i]));
    if (abridged) std::cout << "...";
    printf(" ] - count: %i\n", n);
}
