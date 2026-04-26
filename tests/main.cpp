#include <gtest/gtest.h>

#include <cuda_runtime_api.h>
#include <driver_types.h>

void get_device_properties()
{
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));

    if (device_count == 0) fprintf(stderr, "No CUDA-capable devices found.\n");

    for (int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp device_prop{};
        err = cudaGetDeviceProperties(&device_prop, i);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to get properties for device %i: %s\n", i,
                    cudaGetErrorString(err));
            continue;
        }

        printf("DEVICE %i PROPERTIES:\n", i);
        printf("Name: %s\n", device_prop.name);
        printf("Total Global Memory: %zu bytes\n", device_prop.totalGlobalMem);
        printf("Compute Capability: %i.%i\n", device_prop.major, device_prop.minor);
        printf("Number of Multiprocessors: %i\n", device_prop.multiProcessorCount);
        printf("shared Memory Per Block: %zu bytes\n", device_prop.sharedMemPerBlock);
        printf("Registers Per Block: %i\n", device_prop.regsPerBlock);
        printf("Warp Size: %i\n\n", device_prop.warpSize);
    }
}

int main(int argc, char** argv)
{
    get_device_properties();

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
