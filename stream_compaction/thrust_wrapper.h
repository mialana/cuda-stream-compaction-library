#pragma once

#include "common.h"

namespace stream_compaction::thrust_wrapper
{
struct IsNonZero
{
    __host__ __device__ bool operator()(const int x) const
    { return x != 0; }
};

common::PerformanceTimer& get_timer();

void scan(int n, const int* idata, int* odata);

void radix_sort(int n, const int* idata, int* odata);

void radix_sort_by_key(int n, const int* ikeys, const int* ivalues, int* okeys, int* ovalues);

int compact_by_key(int n, const int* ikeys, const float* ivalues, int* okeys, float* ovalues);

}  // namespace stream_compaction::thrust_wrapper
