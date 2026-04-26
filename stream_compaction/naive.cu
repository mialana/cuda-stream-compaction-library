#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace stream_compaction::naive
{
using stream_compaction::common::eTimerDevice;
using stream_compaction::common::PerformanceTimer;

PerformanceTimer& get_timer()
{
    static PerformanceTimer timer;
    return timer;
}

// in_scan is input and out_scan is output for this iteration
__global__ void kernel_perform_naive_scan_iteration(const int n, const int iter, const int* in_scan,
                                                    int* out_scan)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    int iter_start_idx = 1 << iter - 1;
    if (index >= n)
    {
        return;
    }

    if (index < iter_start_idx)
    {
        out_scan[index] = in_scan[index];
    }
    else
    {
        out_scan[index] = in_scan[index - iter_start_idx] + in_scan[index];
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, const int* idata, int* odata)
{
    // create two device arrays to ping-pong between
    int* dev_idata;
    int* dev_odata;

    cudaMalloc(reinterpret_cast<void**>(&dev_idata), sizeof(int) * n);
    CUDA_CHECK("CUDA malloc for scan array A failed.");

    cudaMalloc(reinterpret_cast<void**>(&dev_odata), sizeof(int) * n);
    CUDA_CHECK("CUDA malloc for scan array B failed.");

    cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    CUDA_CHECK("Memory copy from input data to scan array A failed.");
    cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
    CUDA_CHECK("Memory copy from output data to scan array B failed.");

    cudaDeviceSynchronize();

    get_timer().start_timer<eTimerDevice::GPU>();

    int blocks = divup(n, BLOCK_SIZE);

    for (int i = 1; i <= ilog2_ceil(n); i++)
    {
        kernel_perform_naive_scan_iteration<<<blocks, BLOCK_SIZE>>>(n, i, dev_idata, dev_odata);
        CUDA_CHECK("Perform Naive Scan Iteration CUDA kernel failed.");

        // ping-pong
        int* temp = dev_idata;
        dev_idata = dev_odata;
        dev_odata = temp;
    }

    // result ends up in dev_scanA
    common::kernel_inclusive_to_exclusive<<<blocks, BLOCK_SIZE>>>(n, 0, dev_idata, dev_odata);
    CUDA_CHECK("Inclusive to Exclusive CUDA kernel failed.");

    get_timer().end_timer<eTimerDevice::GPU>();

    cudaMemcpy(odata, dev_odata, sizeof(int) * n,
               cudaMemcpyDeviceToHost);  // result ends up in scanB

    cudaFree(dev_idata);
    cudaFree(dev_odata);  // can't forget memory leaks!
}
}  // namespace stream_compaction::naive
