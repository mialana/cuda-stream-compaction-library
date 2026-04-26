#include <cstdio>
#include "testing_helpers.h"

#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust_wrapper.h>
#include <stream_compaction/radix.h>
#include <stream_compaction/shared.h>

using stream_compaction::common::eTimerDevice;

int* a = new int[kSIZE];
int* b = new int[kSIZE];
int* c = new int[kSIZE];
int* d = new int[kSIZE];
int* e = new int[kSIZE];

int* values = new int[kSIZE];

int* consecutive = new int[kSIZE];  // use to test without randomness
int* consecutive_out = new int[kSIZE];

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

void do_scan_tests()
{
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    // initialize b using stream_compaction::CPU::scan you implement
    // We use b for further comparison. Make sure your stream_compaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.

    zero_array(kSIZE, b);
    print_desc("cpu scan, power-of-two");
    stream_compaction::cpu::scan(kSIZE, a, b);
    stream_compaction::cpu::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::CPU>();
    print_array(kSIZE, b);

    zero_array(kSIZE, c);
    print_desc("cpu scan, non-power-of-two");
    stream_compaction::cpu::scan(kNPOT, a, c);
    stream_compaction::cpu::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::CPU>();
    print_array(kNPOT, c);
    print_cmp_result(kNPOT, b, c);

    zero_array(kSIZE, c);
    print_desc("naive scan, power-of-two");
    stream_compaction::naive::scan(kSIZE, a, c);
    stream_compaction::naive::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    print_array(kSIZE, c);
    print_cmp_result(kSIZE, b, c);

    // For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    ones_array(kSIZE, c);
    print_desc("1s array for finding bugs");
    stream_compaction::naive::scan(kSIZE, a, c);
    print_array(kSIZE, c);

    zero_array(kSIZE, c);
    print_desc("naive scan, non-power-of-two");
    stream_compaction::naive::scan(kNPOT, a, c);
    stream_compaction::naive::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    print_array(kSIZE, c);
    print_cmp_result(kNPOT, b, c);

    zero_array(kSIZE, c);
    print_desc("work-efficient scan, power-of-two");
    stream_compaction::efficient::scan_wrapper(kSIZE, a, c);
    stream_compaction::efficient::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    ;
    print_array(kSIZE, c);
    print_cmp_result(kSIZE, b, c);

    zero_array(kSIZE, c);
    print_desc("work-efficient scan, non-power-of-two");
    stream_compaction::efficient::scan_wrapper(kNPOT, a, c);
    stream_compaction::efficient::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    ;
    print_array(kNPOT, c);
    print_cmp_result(kNPOT, b, c);

    zero_array(kSIZE, c);
    print_desc("work-efficient shared scan, power-of-two");
    stream_compaction::shared::scan_wrapper(kSIZE, c, a);
    stream_compaction::shared::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    print_array(kSIZE, c);
    print_cmp_result(kSIZE, b, c);

    zero_array(kSIZE, c);
    print_desc("work-efficient shared scan, non-power-of-two");
    stream_compaction::shared::scan_wrapper(kNPOT, c, a);
    stream_compaction::shared::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    print_array(kNPOT, c);
    print_cmp_result(kNPOT, b, c);

    zero_array(kSIZE, c);
    print_desc("thrust scan, power-of-two");
    stream_compaction::thrust_wrapper::scan(kSIZE, a, c);
    stream_compaction::thrust_wrapper::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    print_array(kSIZE, c);
    print_cmp_result(kSIZE, b, c);

    zero_array(kSIZE, c);
    print_desc("thrust scan, non-power-of-two");
    stream_compaction::thrust_wrapper::scan(kNPOT, a, c);
    stream_compaction::thrust_wrapper::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    print_array(kNPOT, c);
    print_cmp_result(kNPOT, b, c);
}

void do_consecutive_tests()
{
    zero_array(kSIZE, consecutive_out);
    print_desc("cpu scan, power-of-two, consecutive-valued array");
    stream_compaction::cpu::scan(kSIZE, consecutive, consecutive_out);
    stream_compaction::cpu::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::CPU>();
    print_array(kSIZE, consecutive_out);

    zero_array(kSIZE, c);
    print_desc("work-efficient scan, power-of-two, consecutive-valued array");
    stream_compaction::efficient::scan_wrapper(kSIZE, consecutive, c);
    stream_compaction::efficient::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    ;
    print_array(kSIZE, c);
    print_cmp_result(kSIZE, consecutive_out, c);
}

void do_radix_sort_tests()
{
    printf("\n");
    printf("*****************************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("*****************************\n");

    zero_array(kSIZE, b);
    print_desc("THRUST radix sort, power-of-two");
    stream_compaction::thrust_wrapper::radix_sort(kSIZE, a, b);
    stream_compaction::thrust_wrapper::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    print_array(kSIZE, b);

    zero_array(kSIZE, c);  // want to do in-place
    print_desc("radix sort, power-of-two");
    stream_compaction::radix::sort_wrapper(kSIZE, 6, BLOCK_SIZE, a, c);

    stream_compaction::radix::timer().print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    print_array(kSIZE, c);

    print_cmp_result(kSIZE, b, c);

    zero_array(kSIZE, b);
    zero_array(kSIZE, c);
    zero_array(kSIZE, d);
    zero_array(kSIZE, e);

    print_desc("THRUST radix sort by key, power-of-two");
    stream_compaction::thrust_wrapper::radix_sort_by_key(kSIZE, a, values, b, c);
    stream_compaction::thrust_wrapper::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    print_array(kSIZE, b);
    print_array(kSIZE, c);

    print_desc("custom radix sort by key, power-of-two");
    stream_compaction::radix::sort_by_key_wrapper(kSIZE, 6, BLOCK_SIZE, a, values, d, e);

    stream_compaction::radix::timer().print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    print_array(kSIZE, d);
    print_array(kSIZE, e);

    print_desc("Sorted keys array comparison");
    print_cmp_result(kSIZE, b, d);

    print_desc("Sorted values array comparison");
    print_cmp_result(kSIZE, c, e);
}

void do_stream_compaction_tests()
{
    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    gen_array(kSIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[kSIZE - 1] = 0;
    print_array(kSIZE, a);

    // initialize b using stream_compaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your stream_compaction::CPU::compactWithoutScan is correct.
    zero_array(kSIZE, b);
    print_desc("cpu compact without scan, power-of-two");
    int expected_count = stream_compaction::cpu::compact_without_scan(kSIZE, a, b);
    stream_compaction::cpu::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::CPU>();
    print_array(expected_count, b);

    zero_array(kSIZE, c);
    print_desc("cpu compact without scan, non-power-of-two");
    int expected_npot = stream_compaction::cpu::compact_without_scan(kNPOT, a, c);
    stream_compaction::cpu::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::CPU>();

    zero_array(kSIZE, c);
    print_desc("cpu compact with scan");
    int count = stream_compaction::cpu::compact_with_scan(kSIZE, a, c);
    stream_compaction::cpu::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::CPU>();
    print_array(count, c);
    print_cmp_result(expected_count, b, c);

    zero_array(kSIZE, c);
    print_desc("cpu compact with scan, non-power-of-two");
    count = stream_compaction::cpu::compact_with_scan(kNPOT, a, c);
    stream_compaction::cpu::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::CPU>();
    print_array(count, c);
    print_cmp_result(expected_npot, b, c);

    zero_array(kSIZE, c);
    print_desc("work-efficient shared compact, power-of-two");
    count = stream_compaction::shared::compact_wrapper(kSIZE, a, c);
    stream_compaction::shared::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    print_array(expected_count, c);
    print_cmp_result(expected_count, b, c);

    zero_array(kSIZE, c);
    print_desc("work-efficient shared compact, non-power-of-two");
    count = stream_compaction::shared::compact_wrapper(kNPOT, a, c);
    stream_compaction::shared::get_timer()
        .print_elapsed_time_for_previous_operation<eTimerDevice::GPU>();
    print_array(count, c);
    print_cmp_result(expected_npot, b, c);
}

int main()
{
    get_device_properties();

    print_desc("a array (input)");
    gen_array(kSIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[kSIZE - 1] = 0;
    print_array(kSIZE, a);

    print_desc("values array (input)");
    gen_array(kSIZE - 1, values, 50);  // Leave a 0 at the end to test that edge case
    values[kSIZE - 1] = 0;
    print_array(kSIZE, values);

    print_desc("consecutive array (input)");
    gen_consecutive_array(kSIZE, consecutive);
    print_array(kSIZE, consecutive);

    do_scan_tests();

    do_consecutive_tests();

    do_radix_sort_tests();

    do_stream_compaction_tests();

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
    delete[] consecutive_out;
}
