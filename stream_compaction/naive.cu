#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace stream_compaction::naive
{
using enum common::eTimerDevice;
using common::PerformanceTimer;

PerformanceTimer& get_timer()
{
    static PerformanceTimer timer;
    return timer;
}

__global__ void kernel_hillis_steele_scan(int n, int stride, const int* idata, int* odata)
{
    int index = common::kernel_compute_global_index_1d();

    if (index >= n) return;

    if (index < stride) odata[index] = idata[index];
    else odata[index] = idata[index - stride] + idata[index];
}

void scan(int n, int block_size, int*& dev_idata, int*& dev_odata)
{
    get_timer().start_timer<GPU>();

    int blocks = divup(n, block_size);

    for (int iter = 1; iter <= ilog2_ceil(n); ++iter)
    {
        int stride = 1 << (iter - 1);
        kernel_hillis_steele_scan<<<blocks, block_size>>>(n, stride, dev_idata, dev_odata);
        CUDA_KERNEL_CHECK();

        // ping-pong. latest data always ends up in `dev_idata`
        std::swap(dev_idata, dev_odata);
    }

    // convert to an exclusive kernel
    common::kernel_inclusive_to_exclusive<<<blocks, block_size>>>(n, 0, dev_idata, dev_odata);
    CUDA_KERNEL_CHECK();

    get_timer().end_timer<GPU>();
}

void scan_wrapper(int n, int block_size, const int* idata, int* odata)
{
    // create two device arrays to ping-pong between
    int* dev_idata;
    int* dev_odata;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_idata), sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_odata), sizeof(int) * n));

    CUDA_CHECK(cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

    scan(n, block_size, dev_idata, dev_odata);

    CUDA_CHECK(cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost));

    // free memory
    cudaFree(dev_idata);
    cudaFree(dev_odata);
}
}  // namespace stream_compaction::naive
