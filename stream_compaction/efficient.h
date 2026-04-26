#pragma once

#include "common.h"

namespace stream_compaction::efficient
{
stream_compaction::common::PerformanceTimer& get_timer();

void scan(int n, int block_size, int* dev_scan);

void scan_wrapper(int n, const int* idata, int* odata);

int compact(int n, const int* idata, int* odata);
}  // namespace stream_compaction::efficient
