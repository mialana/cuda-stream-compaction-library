#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

using stream_compaction::common::eTimerDevice;
using stream_compaction::common::PerformanceTimer;

PerformanceTimer& stream_compaction::efficient::get_timer()
{
    static PerformanceTimer timer;
    return timer;
}

__global__ void kernel_efficient_up_sweep(const unsigned long long padded_n, const int stride,
                                          const int prev_stride, int* scan)
{
    unsigned stride_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 0, 1, 2, 3... (like normal)
    // but this is not target elem index

    unsigned long long stride_start = stride_idx * stride;  // index where this stride starts

    // last index in stride. accumulated value of stride always goes here
    unsigned long long accumulator_idx = stride_start + stride - 1;

    if (accumulator_idx >= padded_n)
    {
        return;
    }

    int accumulator = scan[accumulator_idx];  // pre-fetch accumulator's value

    // this new stride has swallowed two strides total
    // siblingIdx is the index of the other stride that now no longer exists
    unsigned long long sibling_idx = stride_start + prev_stride
                                     - 1;  // doesn't depend on accumulator

    scan[accumulator_idx] = accumulator + scan[sibling_idx];
}

__global__ void kernel_efficient_down_sweep(const unsigned long long padded_n, const int stride,
                                            const int next_stride,  // nextStride == (stride / 2)
                                            int* scan)
{
    unsigned stride_idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long stride_start = stride_idx * stride;

    unsigned long long right_child_idx = stride_start + stride - 1;
    if (right_child_idx >= padded_n)
    {
        return;
    }

    int right_child = scan[right_child_idx];

    // leftChild and rightChild are nextSTRIDE indices apart
    unsigned long long left_child_idx = stride_start + next_stride - 1;
    int left_child = scan[left_child_idx];  // does not depend on first memory read

    // give left child right child's value
    // its value has not changed since the end of upsweep
    // it has it easier than right child.
    // on this update it now has accumulated vals of all strides of size next_stride, besides its own
    scan[left_child_idx] = right_child;  // depends on first read, but not second

    // right child currently contains accumulated vals of all strides of size stride besodes its own
    // adding the left child, which only contains values of one stride of size next_stride
    // means that right child now also has accumulated vals of all strides of size next_stride
    // besides its own (same status as left_child)
    scan[right_child_idx] = right_child + left_child;  // memory writes do not depend on each other

    // summary: at each layer, the updated elements get the value of all strides of size next_stride
    // besides its own.
    // so when next_stride == 1, then this element is done, and so are our iterations
}

namespace stream_compaction::efficient
{
/*
    the inner operation of scan without timers and allocation.
    note: dev_scan should be pre-allocated to the padded power of two size
*/
void scan(int n, const int block_size, int* dev_scan)
{
    int num_layers = ilog2_ceil(n);
    int padded_n = 1 << num_layers;  // pad to nearest power of 2

    int prev_stride = 1;  // 1, 2, 4, 8, ... n/2
    int stride = 2;  // essentially the amount of indices that are accumulated into 1 at this iter
    // 2, 4, 8, ... n
    for (int iter = 0; iter < num_layers; iter++)
    {
        // n/2, n/4, n/8, ... 1
        unsigned blocks = divup(padded_n >> (iter + 1), BLOCK_SIZE);
        kernel_efficient_up_sweep<<<blocks, BLOCK_SIZE>>>(padded_n, stride, prev_stride, dev_scan);
        CUDA_CHECK("Perform Work-Efficient Scan Up Sweep Iteration CUDA kernel failed.");

        prev_stride = stride;
        stride = stride <<= 1;
    }

    // set last value of dev_scan to 0
    int replacement = 0;
    cudaMemcpy(&dev_scan[padded_n - 1], &replacement, sizeof(int), cudaMemcpyHostToDevice);

    stride = static_cast<int>(padded_n);  // n, n/2, n/4, ... 2
    int next_stride = stride >> 1;  // n/2, n/4, ... 1
    for (int iter = num_layers; iter > 0; iter--)
    {
        unsigned blocks = divup(padded_n >> iter, BLOCK_SIZE);
        kernel_efficient_down_sweep<<<blocks, BLOCK_SIZE>>>(padded_n, stride, next_stride, dev_scan);
        CUDA_CHECK("Perform Work-Efficient Scan Down Sweep Iteration CUDA kernel failed.");

        stride = next_stride;
        next_stride >>= 1;  // n/2, n/4, n/8, n/16, ...
    }
}

/************************************************************************************************ */

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan_wrapper(int n, const int* idata, int* odata)
{
    int padded_n = 1 << ilog2_ceil(n);

    // create two device arrays
    int* dev_scan;

    cudaMalloc(reinterpret_cast<void**>(&dev_scan), sizeof(int) * padded_n);
    CUDA_CHECK("CUDA malloc for scan array failed.");

    cudaMemcpy(dev_scan, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    CUDA_CHECK("Memory copy from input data to scan array failed.");

    cudaDeviceSynchronize();

    bool using_timer = false;
    if (!get_timer().gpu_timer_started)  // added in order to call `scan` from other functions.
    {
        get_timer().start_timer<eTimerDevice::GPU>();
        using_timer = true;
    }

    scan(n, BLOCK_SIZE, dev_scan);

    if (using_timer)
    {
        get_timer().end_timer<eTimerDevice::GPU>();
    }

    cudaMemcpy(odata, dev_scan, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(dev_scan);  // can't forget memory leaks!
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int* odata, const int* idata)
{
    // TODO: these arrays are unnecessary. will optimize soon.

    // create device arrays
    int* dev_idata;
    int* dev_odata;

    int* dev_bools;
    int* dev_indices;

    cudaMalloc(reinterpret_cast<void**>(&dev_idata), sizeof(int) * n);
    CUDA_CHECK("CUDA malloc for idata array failed.");

    cudaMalloc(reinterpret_cast<void**>(&dev_odata), sizeof(int) * n);
    CUDA_CHECK("CUDA malloc for odata array failed.");

    cudaMalloc(reinterpret_cast<void**>(&dev_bools), sizeof(int) * n);
    CUDA_CHECK("CUDA malloc for bools array failed.");

    cudaMalloc(reinterpret_cast<void**>(&dev_indices), sizeof(int) * n);
    CUDA_CHECK("CUDA malloc for indices array failed.");

    cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    CUDA_CHECK("Memory copy from input data to idata array failed.");
    cudaMemcpy(dev_bools, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
    CUDA_CHECK("Memory copy from output data to odata array failed.");

    cudaDeviceSynchronize();

    int* indices = new int[n];  // create cpu side indices array
    int* bools = new int[n];

    stream_compaction::efficient::get_timer().start_timer<eTimerDevice::GPU>();

    int blocks = divup(n, BLOCK_SIZE);

    // reuse dev_idata for bools
    common::kernel_map_to_boolean<<<blocks, BLOCK_SIZE>>>(n, dev_idata, dev_bools);

    cudaMemcpy(bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
    CUDA_CHECK("Memory copy from device bools to indices array failed.");

    scan_wrapper(n, indices, bools);

    cudaMemcpy(dev_indices, indices, sizeof(int) * n, cudaMemcpyHostToDevice);
    CUDA_CHECK("Memory copy from indices to device indices array failed.");

    common::kernel_scatter<<<blocks, BLOCK_SIZE>>>(n, dev_bools, dev_indices, dev_idata, dev_odata);

    stream_compaction::efficient::get_timer().end_timer<eTimerDevice::GPU>();

    cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(dev_idata);
    cudaFree(dev_odata);
    cudaFree(dev_bools);
    cudaFree(dev_indices);

    return indices[n - 1] + bools[n - 1];
}
}  // namespace stream_compaction::efficient
