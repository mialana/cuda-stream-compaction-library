#pragma once

#include "common.h"

namespace stream_compaction::cpu
{
stream_compaction::common::PerformanceTimer& get_timer();

void scan(int n, const int* idata, int* odata);

int compact_without_scan(int n, const int* idata, int* odata);

int compact_with_scan(int n, const int* idata, int* odata);
}  // namespace stream_compaction::cpu
