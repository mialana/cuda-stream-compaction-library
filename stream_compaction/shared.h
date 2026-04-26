#pragma once

#include "common.h"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

namespace stream_compaction::shared
{

common::PerformanceTimer& get_timer();

__device__ __host__ unsigned kernel_offset(unsigned idx);

__global__ void kernel_scan_intra_block_shared(int padded_n, const int* idata, int* out_block_sums,
                                               int* odata);

__global__ void kernel_add_block_sums(int n, const int* in_block_sums, int* odata);

void scan(int n, int block_size, int* dev_block_sums, const int* dev_idata, int* dev_odata);

void scan_wrapper(int n, int* odata, const int* idata);

int compact(int n, int block_size, const int* dev_idata, int* dev_bools, int* dev_indices,
            int* dev_block_sums, int* dev_odata);

int compact_wrapper(int n, const int* idata, int* odata);

int compact_by_key(int n, int block_size, const int* dev_idata, const int* dev_ivalues,
                   int* dev_indices, int* dev_block_sums, int* dev_bools, int* dev_odata,
                   int* dev_ovalues);

// Host wrapper for compact_by_key. It accepts host arrays for values and keys.
// For example, ivalues and ikeys are input arrays,
// ovalues and okeys will receive the compacted results.
// The function returns the number of surviving (compacted) elements.
int compact_by_key_wrapper(int n, const int* ikeys, const int* ivalues, int* okeys, int* ovalues);

}  // namespace stream_compaction::shared
