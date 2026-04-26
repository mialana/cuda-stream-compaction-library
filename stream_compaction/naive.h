#pragma once

#include "common.h"

namespace stream_compaction::naive
{
stream_compaction::common::PerformanceTimer& get_timer();

void scan(int n, const int* idata, int* odata);
}  // namespace stream_compaction::naive
