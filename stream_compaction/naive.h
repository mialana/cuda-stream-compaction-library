#pragma once

#include "common.h"

namespace StreamCompaction
{
namespace Naive
{
StreamCompaction::Common::PerformanceTimer& get_timer();

void scan(int n, int* odata, const int* idata);
}  // namespace Naive
}  // namespace StreamCompaction
