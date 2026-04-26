#pragma once

#include "common.h"

namespace stream_compaction::naive
{
common::PerformanceTimer& get_timer();

__global__ void kernel_perform_naive_scan_iteration(int n, int iter, const int* idata, int* odata);

void scan(int n, const int* idata, int* odata);
}  // namespace stream_compaction::naive
