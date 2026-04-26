#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "shared.h"

namespace stream_compaction::shared
{

using enum common::eTimerDevice;
using common::PerformanceTimer;

PerformanceTimer& shared::get_timer()
{
    static PerformanceTimer timer;
    return timer;
}

__device__ __host__ unsigned kernel_offset(unsigned idx)
{
    return idx + CONFLICT_FREE_OFFSET(idx);
}

__global__ void kernel_scan_intra_block_shared(int padded_n, const int* idata, int* out_block_sums,
                                               int* odata)
{
    extern __shared__ int mat[];

    const unsigned tile_size = blockDim.x * 2;

    const unsigned tid = threadIdx.x;

    unsigned block_offset = (blockIdx.x * blockDim.x) * 2;
    unsigned thread_offset = 2 * tid;  // first index this thread is responsible for

    unsigned global_thread_idx = block_offset + thread_offset;

    // global memory is read from in coalesced fashion
    // ensure some threads do not return early without zero-padding the shared matrix
    mat[kernel_offset(thread_offset)] = (global_thread_idx < padded_n)
                                            ? idata[block_offset + thread_offset]
                                            : 0;
    mat[kernel_offset(thread_offset + 1)] = (global_thread_idx + 1 < padded_n)
                                                ? idata[block_offset + thread_offset + 1]
                                                : 0;

    // which stride each child is reponsible for -- constant per thread
    // in reality, it is one stride higher than expected, but that's due to -1
    const unsigned stride_idx_first_child = thread_offset + 1;
    const unsigned stride_idx_second_child = thread_offset + 2;

    int stride = 1;  // 1, 2, 4, 8, 16, 32, ... tileSize
    // activeThreads: n/2, n/4, n/8, ... 1
    for (unsigned active_threads = tile_size >> 1; active_threads > 0; active_threads >>= 1)
    {
        __syncthreads();

        if (tid < active_threads)
        {
            unsigned first_idx = stride_idx_first_child * stride - 1;
            unsigned second_idx = stride_idx_second_child * stride - 1;

            mat[kernel_offset(second_idx)] += mat[kernel_offset(first_idx)];
        }
        stride *= 2;
    }

    __syncthreads();

    if (tid == 0)
    {
        out_block_sums[blockIdx.x]
            = mat[kernel_offset(tile_size - 1)];  // write accumulated val of block
        mat[kernel_offset(tile_size - 1)] = 0;  // clear last element
    }

    for (unsigned active_threads = 1; active_threads < tile_size; active_threads <<= 1)
    {
        stride >>= 1;  // STRIDE ended at tileSize
        __syncthreads();

        if (tid < active_threads)
        {
            unsigned first_idx = stride_idx_first_child * stride - 1;
            unsigned second_idx = stride_idx_second_child * stride - 1;

            first_idx = kernel_offset(first_idx);
            second_idx = kernel_offset(second_idx);
            int temp = mat[first_idx];
            mat[first_idx] = mat[second_idx];
            mat[second_idx] += temp;
        }
    }

    __syncthreads();  // this time the last `__syncthreads()` wasn't called

    if (global_thread_idx < padded_n)
    {
        odata[block_offset + thread_offset] = mat[kernel_offset(thread_offset)];
    }
    if (global_thread_idx + 1 < padded_n)
    {
        odata[block_offset + thread_offset + 1] = mat[kernel_offset(thread_offset + 1)];
    }
}

__global__ void kernel_add_block_sums(int n, const int* in_block_sums, int* odata)
{
    __shared__ int block_offset;

    if (threadIdx.x == 0)
    {
        block_offset = in_block_sums[blockIdx.x];
    }

    __syncthreads();

    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
    {  // should be safe to return now
        return;
    }

    odata[index] += block_offset;
}

/*
    the inner operation of scan without timers and allocation.
    note: dev_scan should be pre-allocated to the padded power of two size
*/
void scan(int n, int block_size, int* dev_block_sums, const int* dev_idata, int* dev_odata)
{
    int padded_n = 1 << ilog2_ceil(n);  // pad to nearest power of 2

    const int block_span = block_size * 2;
    // perform scan on the block level
    int num_blocks = divup(padded_n, block_span);

    // numBlocks, numThreads, shared mem size
    kernel_scan_intra_block_shared<<<num_blocks, block_size,
                                     kernel_offset(block_span) * sizeof(int)>>>(padded_n, dev_idata,
                                                                                dev_block_sums,
                                                                                dev_odata);
    CUDA_KERNEL_CHECK();

    if (num_blocks > 1)
    {
        // Allocate temporary buffer for recursive scan of block sums
        int* dev_new_o_data;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_new_o_data), sizeof(int) * num_blocks));

        // Recursively scan the block sums
        scan(num_blocks, block_size, dev_block_sums, dev_block_sums, dev_new_o_data);

        // Add the recursively scanned block sums to the output
        kernel_add_block_sums<<<num_blocks, block_span>>>(padded_n, dev_new_o_data, dev_odata);

        // Free the temporary buffer
        cudaFree(dev_new_o_data);
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan_wrapper(int n, int* odata, const int* idata)
{
    int padded_n = 1 << ilog2_ceil(n);

    int total_blocks = divup(padded_n, 2 * kBLOCK_SIZE);

    // create two device arrays
    int* dev_idata;
    int* dev_odata;
    int* dev_block_sums;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_idata), sizeof(int) * padded_n));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_odata), sizeof(int) * padded_n));

    // create new array to store total sum of each block
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_block_sums), sizeof(int) * total_blocks));

    CUDA_CHECK(cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

    bool using_timer = false;
    if (!get_timer().gpu_timer_started)  // added in order to call `scan` from other functions.
    {
        get_timer().start_timer<GPU>();
        using_timer = true;
    }

    scan(n, kBLOCK_SIZE, dev_block_sums, dev_idata, dev_odata);

    if (using_timer)
    {
        get_timer().end_timer<GPU>();
    }

    CUDA_CHECK(cudaMemcpy(odata, dev_odata, sizeof(int) * n,
                          cudaMemcpyDeviceToHost));  // only copy n elements

    cudaFree(dev_idata);  // can't forget memory leaks!
    cudaFree(dev_odata);
    cudaFree(dev_block_sums);
}

int compact(int n, int block_size, const int* dev_idata, int* dev_bools, int* dev_indices,
            int* dev_block_sums, int* dev_odata)
{
    int blocks = divup(n, block_size);

    common::kernel_map_to_boolean<<<blocks, block_size>>>(n, dev_idata, dev_bools);

    scan(n, block_size, dev_block_sums, dev_bools, dev_indices);

    common::kernel_scatter<<<blocks, block_size>>>(n, dev_bools, dev_indices, dev_idata, dev_odata);

    int last_index;
    int last_bool;

    CUDA_CHECK(cudaMemcpy(&last_index, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_bool, &dev_bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost));

    return last_index + last_bool;
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * Returns the number of surviving elements (i.e. non-zero).
 */
int compact_wrapper(int n, const int* idata, int* odata)
{
    int padded_n = 1 << ilog2_ceil(n);  // pad to nearest power of 2
    int total_blocks = divup(padded_n, 2 * kBLOCK_SIZE);  // for scan block sums

    // Allocate device arrays
    int* dev_idata;
    int* dev_odata;
    int* dev_bools;
    int* dev_indices;
    int* dev_block_sums;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_idata), sizeof(int) * padded_n));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_odata), sizeof(int) * padded_n));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_bools), sizeof(int) * n));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_indices), sizeof(int) * n));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_block_sums), sizeof(int) * total_blocks));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice));

    // Run the compaction and time it
    bool using_timer = false;
    if (!get_timer().gpu_timer_started)
    {
        get_timer().start_timer<GPU>();
        using_timer = true;
    }

    int compact_count = compact(n, kBLOCK_SIZE, dev_idata, dev_bools, dev_indices, dev_block_sums,
                                dev_odata);

    if (using_timer)
    {
        get_timer().end_timer<GPU>();
    }

    // Copy the compacted result back to host; note that compactCount elements are valid
    CUDA_CHECK(cudaMemcpy(odata, dev_odata, sizeof(int) * compact_count, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(dev_idata);
    cudaFree(dev_odata);
    cudaFree(dev_bools);
    cudaFree(dev_indices);
    cudaFree(dev_block_sums);

    return compact_count;
}

int compact_by_key(int n, int block_size, const int* dev_idata, const int* dev_ivalues,
                   int* dev_indices, int* dev_block_sums, int* dev_bools, int* dev_odata,
                   int* dev_ovalues)
{
    int blocks = divup(n, block_size);

    common::kernel_map_to_boolean<<<blocks, block_size>>>(n, dev_idata, dev_bools);

    scan(n, block_size, dev_block_sums, dev_bools, dev_indices);

    common::kernel_scatter<<<blocks, block_size>>>(n, dev_bools, dev_indices, dev_idata, dev_odata);

    cudaDeviceSynchronize();

    common::kernel_scatter<<<blocks, block_size>>>(n, dev_bools, dev_indices, dev_ivalues,
                                                   dev_ovalues);

    int last_index;
    int last_bool;

    CUDA_CHECK(cudaMemcpy(&last_index, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_bool, &dev_bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost));

    return last_index + last_bool;
}

int compact_by_key_wrapper(int n, const int* ikeys, const int* ivalues, int* okeys, int* ovalues)
{
    int *dev_ivalues, *dev_ovalues;
    int *dev_ikeys, *dev_okeys, *dev_bools, *dev_indices, *dev_block_sums;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_ivalues), n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_ovalues), n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_ikeys), n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_okeys), n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_bools), n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_indices), n * sizeof(int)));

    int blocks = divup(n, kBLOCK_SIZE);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_block_sums), blocks * sizeof(int)));

    // Copy input data from host to device.
    CUDA_CHECK(cudaMemcpy(dev_ivalues, ivalues, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_ikeys, ikeys, n * sizeof(int), cudaMemcpyHostToDevice));

    // Call the templated device function from shared.h.
    // (This kernel launches both key and value scatter)
    int count = compact_by_key(n, kBLOCK_SIZE, dev_bools, dev_okeys, dev_ivalues, dev_ovalues,
                               dev_ikeys, dev_block_sums, dev_indices);

    // Copy compacted results back to host.
    CUDA_CHECK(cudaMemcpy(ovalues, dev_ovalues, count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(okeys, dev_okeys, count * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory.
    cudaFree(dev_ivalues);
    cudaFree(dev_ovalues);
    cudaFree(dev_ikeys);
    cudaFree(dev_okeys);
    cudaFree(dev_bools);
    cudaFree(dev_indices);
    cudaFree(dev_block_sums);

    return count;
}
}  // namespace stream_compaction::shared
