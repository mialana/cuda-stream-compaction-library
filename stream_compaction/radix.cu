#include "radix.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "shared.h"

namespace stream_compaction::radix
{

using enum common::eTimerDevice;
using common::PerformanceTimer;

PerformanceTimer& get_timer()
{
    static PerformanceTimer timer;
    return timer;
}

__device__ __host__ int kernel_isolate_bit(int n, int target_bit)
{
    return (n >> target_bit) & 1;
}

__global__ void kernel_split(int n, int target_bit, const int* idata, int* out_not_lsb)
{
    unsigned index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    out_not_lsb[index] = kernel_isolate_bit(idata[index], target_bit) ^ 1;
}

__global__ void kernel_compute_scatter_indices(int n, const int target_bit, const int* scan,
                                               const int* idata, int* indices)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    __shared__ int total_falses;
    if (threadIdx.x == 0)
    {
        total_falses = (kernel_isolate_bit(idata[n - 1], target_bit) ^ 1) + scan[n - 1];
    }

    __syncthreads();  // wait for total_falses

    // if value is 1, we shift right by total falses minus falses before current index
    // if value is 0, we set to position based on how many other falses / 0s come before it
    indices[index] = kernel_isolate_bit(idata[index], target_bit)
                         ? static_cast<int>(index) + (total_falses - scan[index])
                         : scan[index];
}

__global__ void kernel_scatter(int n, const int* indices, const int* idata, int* odata)
{
    unsigned index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    int address = indices[index];
    odata[address] = idata[index];  // Scatter the value to its new position
}

void sort(int n, int max_bit_length, int block_size, int* dev_block_sums, int* dev_indices,
          int* dev_idata, int* dev_odata)
{
    for (int target_bit = 0; target_bit < max_bit_length; target_bit++)
    {
        int blocks = divup(n, block_size);

        // Split data into 0s and 1s based on the target bit
        kernel_split<<<blocks, block_size>>>(n, target_bit, dev_idata, dev_odata);

        // Perform scan on the split results
        shared::scan(n, block_size, dev_block_sums, dev_odata, dev_odata);

        // Scatter data based on the split results
        kernel_compute_scatter_indices<<<blocks, block_size>>>(n, target_bit, dev_odata, dev_idata,
                                                               dev_indices);

        kernel_scatter<<<blocks, block_size>>>(n, dev_indices, dev_idata, dev_odata);

        // Swap buffers (ping-pong)
        int* temp = dev_idata;
        dev_idata = dev_odata;
        dev_odata = temp;
    }
}

void sort_wrapper(int n, int max_bit_length, int block_size, const int* idata, int* odata)
{
    const int padded_n = 1 << ilog2_ceil(n);
    const int block_sums = divup(padded_n, 2 * block_size);

    // Allocate device memory for input/output data and scan
    int* dev_idata;
    int* dev_odata;
    int* dev_block_sums;
    int* dev_indices;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_idata), sizeof(int) * padded_n));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_odata), sizeof(int) * padded_n));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_block_sums), sizeof(int) * block_sums));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_indices), sizeof(int) * n));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice));

    bool using_timer = false;
    if (!get_timer().gpu_timer_started)
    {
        get_timer().start_timer<GPU>();
        using_timer = true;
    }

    sort(n, max_bit_length, block_size, dev_block_sums, dev_indices, dev_idata, dev_odata);

    if (using_timer)
    {
        get_timer().end_timer<GPU>();
    }

    // Copy sorted data back to host
    CUDA_CHECK(cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(dev_idata);
    cudaFree(dev_odata);
    cudaFree(dev_block_sums);
    cudaFree(dev_indices);
}

void sort_by_key(int n, int max_bit_length, int block_size, int* dev_block_sums, int* dev_indices,
                 int* dev_ikeys, int* dev_okeys, int* dev_ivalues, int* dev_ovalues)
{
    for (int target_bit = 0; target_bit < max_bit_length; target_bit++)
    {
        int blocks = divup(n, block_size);

        // Split data into 0s and 1s based on the target bit
        kernel_split<<<blocks, block_size>>>(n, target_bit, dev_ikeys, dev_okeys);

        // Perform scan on the split results
        shared::scan(n, block_size, dev_block_sums, dev_okeys, dev_okeys);

        // Scatter data based on the split results
        kernel_compute_scatter_indices<<<blocks, block_size>>>(n, target_bit, dev_okeys, dev_ikeys,
                                                               dev_indices);

        kernel_scatter<<<blocks, block_size>>>(n, dev_indices, dev_ikeys, dev_okeys);
        kernel_scatter<<<blocks, block_size>>>(n, dev_indices, dev_ivalues, dev_ovalues);

        // Swap buffers (ping-pong)
        int* temp = dev_ikeys;
        dev_ikeys = dev_okeys;
        dev_okeys = temp;

        temp = dev_ivalues;
        dev_ivalues = dev_ovalues;
        dev_ovalues = temp;
    }
}

void sort_by_key_wrapper(int n, int max_bit_length, int block_size, const int* ikeys,
                         const int* ivalues, int* okeys, int* ovalues)
{
    const int padded_n = 1 << ilog2_ceil(n);
    const int block_sums = divup(padded_n, 2 * block_size);

    // Allocate device memory for input/output data and scan
    int* dev_ikeys;
    int* dev_okeys;
    int* dev_ivalues;
    int* dev_ovalues;

    int* dev_block_sums;
    int* dev_indices;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_ikeys), sizeof(int) * padded_n));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_okeys), sizeof(int) * padded_n));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_ivalues), sizeof(int) * padded_n));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_ovalues), sizeof(int) * padded_n));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_block_sums), sizeof(int) * block_sums));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_indices), sizeof(int) * n));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(dev_ikeys, ikeys, sizeof(int) * n, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev_ivalues, ivalues, sizeof(int) * n, cudaMemcpyHostToDevice));

    bool using_timer = false;
    if (!get_timer().gpu_timer_started)
    {
        get_timer().start_timer<GPU>();
        using_timer = true;
    }

    sort_by_key(n, max_bit_length, block_size, dev_block_sums, dev_indices, dev_ikeys, dev_okeys,
                dev_ivalues, dev_ovalues);

    if (using_timer)
    {
        get_timer().end_timer<GPU>();
    }

    // Copy sorted data back to host
    CUDA_CHECK(cudaMemcpy(okeys, dev_ikeys, sizeof(int) * n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(ovalues, dev_ivalues, sizeof(int) * n, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(dev_ikeys);
    cudaFree(dev_okeys);
    cudaFree(dev_ivalues);
    cudaFree(dev_ovalues);
    cudaFree(dev_block_sums);
    cudaFree(dev_indices);
}
}  // namespace stream_compaction::radix
