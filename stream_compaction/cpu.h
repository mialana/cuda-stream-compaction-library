#pragma once

#include "common.h"

namespace StreamCompaction
{
namespace CPU
{
StreamCompaction::Common::PerformanceTimer& get_timer();

void scan(int n, int* odata, const int* idata);

int compactWithoutScan(int n, int* odata, const int* idata);

int compactWithScan(int n, int* odata, const int* idata);
}  // namespace CPU
}  // namespace StreamCompaction
