#include "common.h"

void check_cuda_error_fn(const char* msg, const char* file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

namespace stream_compaction::common
{

/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernel_map_to_boolean(int n, const int* idata, int* out_bools)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    out_bools[index] = idata[index] == 0 ? 0 : 1;
}

template __global__ void kernel_scatter<int>(int n, const int* bools, const int* indices,
                                             const int* idata, int* odata);

__global__ void kernel_inclusive_to_exclusive(int n, int identity, const int* idata, int* odata)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
    {
        return;
    }
    else if (index == 0)
    {
        odata[index] = identity;
    }
    else
    {
        odata[index] = idata[index - 1];
    }
}

__global__ void kernel_set_device_array_value(int* arr, int index, int value)
{
    arr[index] = value;  // round up to nearest power of two
}

}  // namespace stream_compaction::common
