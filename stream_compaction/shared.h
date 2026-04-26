#pragma once

#include "common.h"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

namespace stream_compaction::shared
{

common::PerformanceTimer& get_timer();

void scan(int n, int block_size, int* dev_block_sums, const int* dev_idata, int* dev_odata);

void scan_wrapper(int n, int* odata, const int* idata);

int compact(int n, int block_size, int* dev_bools, int* dev_block_sums, int* dev_indices,
            const int* dev_idata, int* dev_odata);

int compact_wrapper(int n, const int* idata, int* odata);

template<typename T>
inline int compact_by_key(int n, int* dev_indices, int* dev_block_sums, int* dev_bools,
                          const T* dev_ivalues, const int* dev_idata, T* dev_ovalues,
                          int* dev_odata, int block_size)
{
    int blocks = divup(n, block_size);

    common::kernel_map_to_boolean<<<blocks, block_size>>>(n, dev_idata, dev_bools);

    scan(n, block_size, dev_block_sums, dev_bools, dev_indices);

    common::kernel_scatter<int>
        <<<blocks, block_size>>>(n, dev_bools, dev_indices, dev_idata, dev_odata);

    cudaDeviceSynchronize();

    common::kernel_scatter<T>
        <<<blocks, block_size>>>(n, dev_bools, dev_indices, dev_ivalues, dev_ovalues);

    int last_index;
    int last_bool;

    cudaMemcpy(&last_index, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_bool, &dev_bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost);

    return last_index + last_bool;
}

// Host wrapper for compact_by_key. It accepts host arrays for values and keys.
// For example, ivalues and ikeys are input arrays,
// ovalues and okeys will receive the compacted results.
// The function returns the number of surviving (compacted) elements.
template<typename T>
inline int compact_by_key_wrapper(int n, const int* ikeys, const T* ivalues, int* okeys, T* ovalues)
{
    T *dev_ivalues, *dev_ovalues;
    int *dev_ikeys, *dev_okeys, *dev_bools, *dev_indices, *dev_block_sums;

    // Allocate device memory
    cudaMalloc(reinterpret_cast<void**>(&dev_ivalues), n * sizeof(T));
    cudaMalloc(reinterpret_cast<void**>(&dev_ovalues), n * sizeof(T));
    cudaMalloc(reinterpret_cast<void**>(&dev_ikeys), n * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&dev_okeys), n * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&dev_bools), n * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&dev_indices), n * sizeof(int));

    int blocks = divup(n, BLOCK_SIZE);
    cudaMalloc(reinterpret_cast<void**>(&dev_block_sums), blocks * sizeof(int));

    // Copy input data from host to device.
    cudaMemcpy(dev_ivalues, ivalues, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ikeys, ikeys, n * sizeof(int), cudaMemcpyHostToDevice);

    // Call the templated device function from shared.h.
    // (This kernel launches both key and value scatter)
    int count = stream_compaction::shared::compact_by_key<T>(n, dev_ivalues, dev_ovalues, dev_ikeys,
                                                             dev_okeys, dev_bools, dev_indices,
                                                             dev_block_sums, BLOCK_SIZE);

    // Copy compacted results back to host.
    cudaMemcpy(ovalues, dev_ovalues, count * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(okeys, dev_okeys, count * sizeof(int), cudaMemcpyDeviceToHost);

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
