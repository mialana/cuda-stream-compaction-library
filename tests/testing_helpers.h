#pragma once

#include <cstdio>
#include <iostream>
#include <chrono>
#include <random>

constexpr int SIZE = 1 << 24;
constexpr int NPOT = SIZE - 3;  // Non-Power-Of-Two

constexpr char PINK[] = "\033[1;35m";
constexpr char RED[] = "\033[1;31m";
constexpr char GREEN[] = "\033[1;32m";
constexpr char RESET[] = "\033[0m";

template<std::integral T>
inline bool cmpArrays(int n, T* a, T* b)
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

inline void printDesc(const char* desc)
{
    std::cout << PINK << "=== " << desc << " ===" << RESET << std::endl;
}

template<std::integral T>
inline void printCmpResult(int n, T* a, T* b)
{
    if (cmpArrays(n, a, b)) std::cout << RED << "FAILED";
    else std::cout << GREEN << "PASSED";
    std::cout << RESET << std::endl;
}

inline void zeroArray(int n, int* a)
{
    for (int i = 0; i < n; ++i)
    {
        a[i] = 0;
    }
}

inline void onesArray(int n, int* a)
{
    for (int i = 0; i < n; ++i)
    {
        a[i] = 1;
    }
}

template<std::integral T>
inline void genArray(int n, T* a, int max_val)
{
    std::random_device rd;
    std::mt19937 gen(rd());  // initialize Mersenne Twister engine
    std::uniform_int_distribution<T> distrib(0, max_val);

    for (int i = 0; i < n; ++i)
    {
        a[i] = distrib(gen);
    }
}

template<std::integral T>
inline void genConsecutiveArray(int n, T* a)
{
    for (int i = 0; i < n; ++i)
    {
        a[i] = static_cast<T>(i);
    }
}

template<std::integral T>
inline void copyArray(int n, const T* a, T* out_copy)
{
    memcpy(out_copy, a, n * sizeof(T));
}

template<std::integral T>
inline void printArray(int n, T* a, bool abridged = true)
{
    int max_size = abridged ? std::min(n, 16) : n;

    std::cout << '\t' << "[ ";
    for (int i = 0; i < max_size; ++i)
    {
        printf("%i ", static_cast<int>(a[i]));
    }
    if (abridged) std::cout << "...";
    printf(" ] - count: %i\n", n);
}

template<typename T>
inline void printElapsedTime(T time)
{
    std::cout << '\t' << "ELAPSED TIME: " << time << "ms" << std::endl;
}
