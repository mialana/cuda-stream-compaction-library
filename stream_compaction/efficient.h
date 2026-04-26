#pragma once

#include "common.h"

namespace stream_compaction::efficient
{
common::PerformanceTimer& get_timer();

__global__ void kernel_efficient_up_sweep(unsigned int padded_n, int stride, int prev_stride,
                                          int* scan);

__global__ void kernel_efficient_down_sweep(int padded_n, int stride,
                                            int next_stride,  // next_stride == (stride / 2)
                                            int* scan);

void scan(int n, int block_size, int* dev_scan);

void scan_wrapper(int n, const int* idata, int* odata);

int compact(int n, const int* idata, int* odata);
}  // namespace stream_compaction::efficient
