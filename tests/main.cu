/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include <stream_compaction/shared.h>
#include "testing_helpers.hpp"

// use during development with `#if !SKIP_UNIMPLEMENTED` preprocessor at desired skip point
#define SKIP_UNIMPLEMENTED 1

const int SIZE = 1 << 24;   // feel free to change the size of array
const int NPOT = SIZE - 3;  // Non-Power-Of-Two

int* a = new int[SIZE];
int* b = new int[SIZE];
int* c = new int[SIZE];
int* d = new int[SIZE];
int* e = new int[SIZE];

int* values = new int[SIZE];

int* consecutive = new int[SIZE];  // use to test without randomness
int* consecutiveOut = new int[SIZE];

void getDeviceProperties()
{
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found.\n");
    }

    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, i);
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "Failed to get properties for device %d: %s\n",
                    i,
                    cudaGetErrorString(err));
            continue;
        }

        printf("--- Device %d Properties ---\n", i);
        printf("Name: %s\n", deviceProp.name);
        printf("Total Global Memory: %zu bytes\n", deviceProp.totalGlobalMem);
        printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Number of Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("Shared Memory Per Block: %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("Registers Per Block: %d\n", deviceProp.regsPerBlock);
        printf("Warp Size: %d\n", deviceProp.warpSize);
        printf("\n");
    }
}

void doScanTests()
{
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.

    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(SIZE, b, true);

#if !SKIP_UNIMPLEMENTED
    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    // For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scanWrapper(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scanWrapper(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

#endif

    zeroArray(SIZE, c);
    printDesc("work-efficient shared scan, power-of-two");
    StreamCompaction::Shared::scanWrapper(SIZE, c, a);
    printElapsedTime(StreamCompaction::Shared::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

#if !SKIP_UNIMPLEMENTED

    zeroArray(SIZE, c);
    printDesc("work-efficient shared scan, non-power-of-two");
    StreamCompaction::Shared::scanWrapper(NPOT, c, a);
    printElapsedTime(StreamCompaction::Shared::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

#endif

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

#if !SKIP_UNIMPLEMENTED
    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
#endif
}

void doConsecutiveTests()
{
    zeroArray(SIZE, consecutiveOut);
    printDesc("cpu scan, power-of-two, consecutive-valued array");
    StreamCompaction::CPU::scan(SIZE, consecutiveOut, consecutive);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(SIZE, consecutiveOut, true);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two, consecutive-valued array");
    StreamCompaction::Efficient::scanWrapper(SIZE, c, consecutive);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, consecutiveOut, c);
}

void doRadixSortTests()
{
    printf("\n");
    printf("*****************************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("*****************************\n");

    zeroArray(SIZE, b);
    printDesc("THRUST radix sort, power-of-two");
    StreamCompaction::Thrust::radixSort(SIZE, b, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);  // want to do in-place
    printDesc("radix sort, power-of-two");
    StreamCompaction::Radix::sortWrapper(SIZE, c, a, 6, BLOCK_SIZE);

    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(SIZE, c, true);

    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, b);
    zeroArray(SIZE, c);
    zeroArray(SIZE, d);
    zeroArray(SIZE, e);

    printDesc("THRUST radix sort by key, power-of-two");
    StreamCompaction::Thrust::radixSortByKey(SIZE, b, c, a, values);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(SIZE, b, true);
    printArray(SIZE, c, true);

    printDesc("custom radix sort by key, power-of-two");
    StreamCompaction::Radix::sortByKeyWrapper(SIZE, d, a, e, values, 6, BLOCK_SIZE);

    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(SIZE, d, true);
    printArray(SIZE, e, true);

    printDesc("Sorted keys array comparison");
    printCmpResult(SIZE, b, d);

    printDesc("Sorted values array comparison");
    printCmpResult(SIZE, c, e);
}

void doStreamCompactionTests()
{
    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4, 0);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

#if !SKIP_UNIMPLEMENTED
    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
#endif

    zeroArray(SIZE, c);
    printDesc("work-efficient shared compact, power-of-two");
    count = StreamCompaction::Shared::compactWrapper(SIZE, c, a);
    printElapsedTime(StreamCompaction::Shared::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(expectedCount, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient shared compact, non-power-of-two");
    count = StreamCompaction::Shared::compactWrapper(NPOT, c, a);
    printElapsedTime(StreamCompaction::Shared::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
}

void doCompactByKeyTest()
{
    printf("\n");
    printf("*********************************************\n");
    printf("** STREAM COMPACTION BY KEY TEST (float) **\n");
    printf("*********************************************\n");

    float* flt_values = new float[SIZE];

    printDesc("a array (same as before, input)");
    printArray(SIZE, a, true);

    printDesc("flt_values array (input)");
    genArray(SIZE, flt_values, 5, 2000);
    printArray(SIZE, flt_values, true);

    // Allocate host arrays for output (results).
    float* out_flt_values_thrust = new float[SIZE];
    float* out_flt_values_custom = new float[SIZE];

    zeroArray(SIZE, b);
    printDesc("thrust compact by key");

    int expectedCount = StreamCompaction::Thrust::compactByKey(SIZE,
                                                               b,
                                                               out_flt_values_thrust,
                                                               a,
                                                               flt_values);

    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray<int>(expectedCount, b, true);
    printArray<float>(expectedCount, out_flt_values_thrust, true);

    zeroArray(SIZE, c);
    printDesc("custom compact by key");

    int count = StreamCompaction::Shared::compactByKeyWrapper<float>(SIZE,
                                                                     out_flt_values_custom,
                                                                     c,
                                                                     flt_values,
                                                                     a);

    printElapsedTime(StreamCompaction::Shared::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printArray<int>(count, c, true);
    printArray<float>(count, out_flt_values_custom, true);

    printDesc("keys comparison");
    printCmpLenResult(count, expectedCount, b, c);
    printDesc("values comparison");
    printCmpLenResult(count, expectedCount, out_flt_values_thrust, out_flt_values_custom);

    delete[] out_flt_values_thrust;
    delete[] out_flt_values_custom;
}

int main()
{
    // getDeviceProperties();

    printDesc("a array (input)");
    genArray(SIZE - 1, a, 50, 0);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    printDesc("values array (input)");
    genArray(SIZE - 1, values, 50, 1000);  // Leave a 0 at the end to test that edge case
    values[SIZE - 1] = 0;
    printArray(SIZE, values, true);

    printDesc("consecutive array (input)");
    genConsecutiveArray(SIZE, consecutive);
    printArray(SIZE, consecutive, true);

    // a = consecutive;

    doScanTests();

    doConsecutiveTests();

    doRadixSortTests();

    doStreamCompactionTests();

    doCompactByKeyTest();

#if defined(_WIN32) || defined(_WIN64)  // errors out on linux
    system("pause");                    // stop Win32 console from closing on exit
#endif

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
    delete[] e;
    delete[] values;

    delete[] consecutive;
    delete[] consecutiveOut;
}
