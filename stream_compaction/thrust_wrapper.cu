#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/detail/vector_base.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "common.h"
#include "thrust_wrapper.h"

namespace stream_compaction::thrust_wrapper
{
using enum common::eTimerDevice;
using common::PerformanceTimer;

using thrust::host_vector;

PerformanceTimer& get_timer()
{
    static PerformanceTimer timer;
    return timer;
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, const int* idata, int* odata)
{
    // Copy data from host to device
    thrust::host_vector<int> host_idata(idata, idata + n);  // thrust host vector
    thrust::device_vector<int> dev_idata = host_idata;  // built-in assignment conversion
    thrust::device_vector<int> dev_odata(n);  // for output

    get_timer().start_timer<GPU>();

    thrust::exclusive_scan(dev_idata.begin(), dev_idata.end(), dev_odata.begin());

    get_timer().end_timer<GPU>();

    // copy result back to host
    thrust::copy(dev_odata.begin(), dev_odata.end(), odata);
}

void radix_sort(int n, const int* idata, int* odata)
{
    thrust::device_vector<int> dev_copy(idata, idata + n);

    bool using_timer = false;
    if (!get_timer().gpu_timer_started)
    {
        get_timer().start_timer<GPU>();
        using_timer = true;
    }

    thrust::sort(dev_copy.begin(), dev_copy.end());

    if (using_timer)
    {
        get_timer().end_timer<GPU>();
    }

    thrust::copy(dev_copy.begin(), dev_copy.end(), odata);
}

void radix_sort_by_key(int n, const int* ikeys, const int* ivalues, int* okeys, int* ovalues)
{
    // Wrap raw pointers with Thrust device pointers
    thrust::device_vector<int> dev_ikeys(ikeys, ikeys + n);
    thrust::device_vector<int> dev_ivalues(ivalues, ivalues + n);

    bool using_timer = false;
    if (!get_timer().gpu_timer_started)
    {
        get_timer().start_timer<GPU>();
        using_timer = true;
    }

    // Sort keys and reorder values accordingly
    thrust::sort_by_key(dev_ikeys.begin(), dev_ikeys.end(), dev_ivalues.begin());

    if (using_timer)
    {
        get_timer().end_timer<GPU>();
    }

    // Copy sorted keys and values back to host
    thrust::copy(dev_ikeys.begin(), dev_ikeys.end(), okeys);
    thrust::copy(dev_ivalues.begin(), dev_ivalues.end(), ovalues);
}

/**
 * Stream compaction by key using Thrust.
 * Given n elements in in_keys and in_values, copies each (key,value) pair
 * for which the key is nonzero into out_keys and out_values.
 * Returns the number of surviving elements.
 */
int compact_by_key(int n, const int* ikeys, const float* ivalues, int* okeys, float* ovalues)
{
    // Wrap raw input arrays into Thrust device vectors.
    thrust::device_vector<int> dev_ikeys(ikeys, ikeys + n);
    thrust::device_vector<float> dev_ivalues(ivalues, ivalues + n);

    // Create a zipped iterator over (key, value)
    auto zipped_begin = thrust::make_zip_iterator(
        thrust::make_tuple(dev_ikeys.begin(), dev_ivalues.begin()));
    auto zipped_end = thrust::make_zip_iterator(
        thrust::make_tuple(dev_ikeys.end(), dev_ivalues.end()));

    // Call remove_if: it shifts surviving elements to the front.
    // Remove pairs if key == 0.
    auto new_end = thrust::remove_if(zipped_begin, zipped_end,
                                     [] __device__(const thrust::tuple<int, float>& tup)
                                     { return thrust::get<0>(tup) == 0; });

    // Compute the new count.
    int count = static_cast<int>(thrust::get<0>(new_end - zipped_begin));

    // Copy the surviving keys and values back to host memory.
    thrust::copy(dev_ikeys.begin(), dev_ikeys.begin() + count, okeys);
    thrust::copy(dev_ivalues.begin(), dev_ivalues.begin() + count, ovalues);

    return count;
}

}  // namespace stream_compaction::thrust_wrapper
