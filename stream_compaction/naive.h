#pragma once

#include "common.h"

namespace stream_compaction::naive
{
common::PerformanceTimer& get_timer();

__global__ void kernel_perform_naive_scan_iteration(int n, int iter, const int* idata, int* odata);

void scan(int n, int block_size, int* dev_idata, int* dev_odata);

// copies host `idata` to device and copies device `dev_odata` back to host after
void scan_wrapper(int n, int block_size, const int* idata, int* odata);
}  // namespace stream_compaction::naive
