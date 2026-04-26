#include "cpu.h"

#include "common.h"

namespace stream_compaction::cpu
{
using enum common::eTimerDevice;
using common::PerformanceTimer;

PerformanceTimer& get_timer()
{
    static PerformanceTimer timer;
    return timer;
}

/**
 * CPU scan (prefix sum).
 * For performance analysis, this is supposed to be a simple for loop.
 * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan
 * in this function first.
 */
void scan(int n, const int* idata, int* odata)
{
    get_timer().start_timer<CPU>();

    int prev_sum = 0;  // save prev sum for access ease
    for (int j = 0; j < n; j++)
    {
        odata[j] = prev_sum;
        prev_sum += idata[j];
    }

    get_timer().end_timer<CPU>();
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compact_without_scan(int n, const int* idata, int* odata)
{
    get_timer().start_timer<CPU>();

    int out_index = 0;  // pointer to current progress in out array

    for (int i = 0; i < n; i++)
    {
        int in_val = idata[i];
        if (in_val != 0)
        {
            odata[out_index] = in_val;
            out_index++;
        }
    }

    get_timer().end_timer<CPU>();
    return out_index;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compact_with_scan(int n, const int* idata, int* odata)
{
    get_timer().start_timer<CPU>();

    int* is_not_zero = new int[n];
    int* scan_is_not_zero = new int[n];

    for (int i = 0; i < n; i++)
    {
        is_not_zero[i] = idata[i] != 0 ? 1 : 0;  // val is 1 at i if idata[i] != 0, else 0
    }

    scan(n, is_not_zero, scan_is_not_zero);  // scan result is index in final array

    for (int i = 0; i < n; i++)
    {
        if (is_not_zero[i])
        {
            odata[scan_is_not_zero[i]] = idata[i];
        }
    }

    get_timer().end_timer<CPU>();

    return scan_is_not_zero[n - 1] + is_not_zero[n - 1];  // due to exclusive scan
}
}  // namespace stream_compaction::cpu
