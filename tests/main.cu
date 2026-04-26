#include <cstdio>
#include "testing_helpers.h"

#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include <stream_compaction/shared.h>

using StreamCompaction::Common::TimerDevice;

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
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));

    if (deviceCount == 0) fprintf(stderr, "No CUDA-capable devices found.\n");

    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, i);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to get properties for device %i: %s\n", i,
                    cudaGetErrorString(err));
            continue;
        }

        printf("DEVICE %i PROPERTIES:\n", i);
        printf("Name: %s\n", deviceProp.name);
        printf("Total Global Memory: %zu bytes\n", deviceProp.totalGlobalMem);
        printf("Compute Capability: %i.%i\n", deviceProp.major, deviceProp.minor);
        printf("Number of Multiprocessors: %i\n", deviceProp.multiProcessorCount);
        printf("Shared Memory Per Block: %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("Registers Per Block: %i\n", deviceProp.regsPerBlock);
        printf("Warp Size: %i\n\n", deviceProp.warpSize);
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
    StreamCompaction::CPU::get_timer().print_elapsed_time_for_previous_operation<TimerDevice::CPU>();
    printArray(SIZE, b);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    StreamCompaction::CPU::get_timer().print_elapsed_time_for_previous_operation<TimerDevice::CPU>();
    printArray(NPOT, c);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    StreamCompaction::Naive::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    printArray(SIZE, c);
    printCmpResult(SIZE, b, c);

    // For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    StreamCompaction::Naive::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    printArray(SIZE, c);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scanWrapper(SIZE, c, a);
    StreamCompaction::Efficient::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    ;
    printArray(SIZE, c);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scanWrapper(NPOT, c, a);
    StreamCompaction::Efficient::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    ;
    printArray(NPOT, c);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient shared scan, power-of-two");
    StreamCompaction::Shared::scanWrapper(SIZE, c, a);
    StreamCompaction::Shared::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    printArray(SIZE, c);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient shared scan, non-power-of-two");
    StreamCompaction::Shared::scanWrapper(NPOT, c, a);
    StreamCompaction::Shared::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    printArray(NPOT, c);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    StreamCompaction::Thrust::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    printArray(SIZE, c);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    StreamCompaction::Thrust::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    printArray(NPOT, c);
    printCmpResult(NPOT, b, c);
}

void doConsecutiveTests()
{
    zeroArray(SIZE, consecutiveOut);
    printDesc("cpu scan, power-of-two, consecutive-valued array");
    StreamCompaction::CPU::scan(SIZE, consecutiveOut, consecutive);
    StreamCompaction::CPU::get_timer().print_elapsed_time_for_previous_operation<TimerDevice::CPU>();
    printArray(SIZE, consecutiveOut);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two, consecutive-valued array");
    StreamCompaction::Efficient::scanWrapper(SIZE, c, consecutive);
    StreamCompaction::Efficient::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    ;
    printArray(SIZE, c);
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
    StreamCompaction::Thrust::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    printArray(SIZE, b);

    zeroArray(SIZE, c);  // want to do in-place
    printDesc("radix sort, power-of-two");
    StreamCompaction::Radix::sortWrapper(SIZE, c, a, 6, BLOCK_SIZE);

    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation());
    printArray(SIZE, c);

    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, b);
    zeroArray(SIZE, c);
    zeroArray(SIZE, d);
    zeroArray(SIZE, e);

    printDesc("THRUST radix sort by key, power-of-two");
    StreamCompaction::Thrust::radixSortByKey(SIZE, b, c, a, values);
    StreamCompaction::Thrust::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    printArray(SIZE, b);
    printArray(SIZE, c);

    printDesc("custom radix sort by key, power-of-two");
    StreamCompaction::Radix::sortByKeyWrapper(SIZE, d, a, e, values, 6, BLOCK_SIZE);

    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation());
    printArray(SIZE, d);
    printArray(SIZE, e);

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

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a);

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    int expected_count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    StreamCompaction::CPU::get_timer().print_elapsed_time_for_previous_operation<TimerDevice::CPU>();
    printArray(expected_count, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    int expected_npot = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    StreamCompaction::CPU::get_timer().print_elapsed_time_for_previous_operation<TimerDevice::CPU>();

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    int count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    StreamCompaction::CPU::get_timer().print_elapsed_time_for_previous_operation<TimerDevice::CPU>();
    printArray(count, c);
    printCmpResult(expected_count, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithScan(NPOT, c, a);
    StreamCompaction::CPU::get_timer().print_elapsed_time_for_previous_operation<TimerDevice::CPU>();
    printArray(count, c);
    printCmpResult(expected_npot, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient shared compact, power-of-two");
    count = StreamCompaction::Shared::compactWrapper(SIZE, c, a);
    StreamCompaction::Shared::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    printArray(expected_count, c);
    printCmpResult(expected_count, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient shared compact, non-power-of-two");
    count = StreamCompaction::Shared::compactWrapper(NPOT, c, a);
    StreamCompaction::Shared::get_timer()
        .print_elapsed_time_for_previous_operation<TimerDevice::GPU>();
    printArray(count, c);
    printCmpResult(expected_npot, b, c);
}

int main()
{
    getDeviceProperties();

    printDesc("a array (input)");
    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a);

    printDesc("values array (input)");
    genArray(SIZE - 1, values, 50);  // Leave a 0 at the end to test that edge case
    values[SIZE - 1] = 0;
    printArray(SIZE, values);

    printDesc("consecutive array (input)");
    genConsecutiveArray(SIZE, consecutive);
    printArray(SIZE, consecutive);

    doScanTests();

    doConsecutiveTests();

    doRadixSortTests();

    doStreamCompactionTests();

#if defined(_WIN32) || defined(_WIN64)  // errors out on linux
    system("pause");  // stop Win32 console from closing on exit
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
