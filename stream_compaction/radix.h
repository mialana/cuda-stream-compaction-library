#pragma once

#include "common.h"

namespace stream_compaction
{
namespace radix
{
common::PerformanceTimer& get_timer();

__device__ __host__ int kernel_isolate_bit(int n, int target_bit);

__global__ void kernel_split(int n, int target_bit, const int* idata, int* out_not_lsb);

__global__ void kernel_compute_scatter_indices(int n, int target_bit, const int* scan,
                                               const int* idata, int* out_indices);

__global__ void kernel_scatter(int n, const int* addresses, const int* idata, int* odata);

void sort(int n, int max_bit_length, int block_size, int* dev_block_sums, int* dev_indices,
          int* dev_idata, int* dev_odata);

void sort_wrapper(int n, int max_bit_length, int block_size, const int* idata, int* odata);

void sort_by_key(int n, int max_bit_length, int block_size, int* dev_block_sums, int* dev_indices,
                 int* dev_ikeys, int* dev_okeys, int* dev_ivalues, int* dev_ovalues);

void sort_by_key_wrapper(int n, int max_bit_length, int block_size, const int* ikeys,
                         const int* ivalues, int* okeys, int* ovalues);

}  // namespace radix
}  // namespace stream_compaction
