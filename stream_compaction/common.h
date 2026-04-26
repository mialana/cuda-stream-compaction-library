#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <iostream>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CUDA_CHECK(msg) check_CUDA_error_fn(msg, FILENAME, __LINE__)

#define BLOCK_SIZE 128

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void check_CUDA_error_fn(const char* msg, const char* file = nullptr, int line = -1);

inline unsigned divup(unsigned size, unsigned div)
{
    return (size + div - 1) / div;
}

inline int ilog2(int x)
{
    int lg = 0;
    while (x >>= 1)
    {
        ++lg;
    }
    return lg;
}

// calculates smallest possible integer k such that 2^k >= x
// subtracts x from 1 in the case that we already have a power of 2
inline int ilog2ceil(int x)
{
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}

namespace StreamCompaction
{
namespace Common
{
__global__ void kernMapToBoolean(int n, int* bools, const int* idata);

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
template<typename T>
__global__ void kernScatter(int n, T* odata, const T* idata, const int* bools, const int* indices)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    if (bools[index] == 1)
    {
        odata[indices[index]] = idata[index];
    }
}

__global__ void kernel_inclusiveToExclusive(int n, int identity, const int* iData, int* oData);

__global__ void kernel_setDeviceArrayValue(int* arr, const int index, const int value);

enum class TimerDevice
{
    CPU,
    GPU
};

/**
 * This class is used for timing the performance
 * Uncopyable and unmovable
 *
 * Adapted from WindyDarian(https://github.com/WindyDarian)
 */
class PerformanceTimer
{
public:
    bool cpu_timer_started = false;
    bool gpu_timer_started = false;

    PerformanceTimer()
    {
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_end);
    }

    ~PerformanceTimer()
    {
        cudaEventDestroy(event_start);
        cudaEventDestroy(event_end);
    }

    template<TimerDevice D>
    void start_timer()
    {
        if constexpr (D == TimerDevice::CPU)
        {
            if (cpu_timer_started) throw std::runtime_error("CPU timer already started");
            cpu_timer_started = true;

            time_start_cpu = std::chrono::high_resolution_clock::now();
        }
        else {
            if (gpu_timer_started) throw std::runtime_error("GPU timer already started");
            gpu_timer_started = true;

            cudaEventRecord(event_start);
        }
    }

    template<TimerDevice D>
    void end_timer()
    {
        if constexpr (D == TimerDevice::CPU)
        {
            time_end_cpu = std::chrono::high_resolution_clock::now();

            if (!cpu_timer_started) throw std::runtime_error("CPU timer not started");

            std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
            prev_elapsed_time_cpu_milliseconds
                = static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

            cpu_timer_started = false;
        }
        else {
            cudaEventRecord(event_end);
            cudaEventSynchronize(event_end);

            if (!gpu_timer_started) throw std::runtime_error("GPU timer not started");

            cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
            gpu_timer_started = false;
        }
    }

    template<TimerDevice D>
    inline void print_elapsed_time_for_previous_operation() const
    {
        float elapsed_time;
        const char* timer_device_string;
        if constexpr (D == TimerDevice::CPU)
        {
            elapsed_time = prev_elapsed_time_cpu_milliseconds;
            timer_device_string = "CPU";
        }
        else {
            elapsed_time = prev_elapsed_time_gpu_milliseconds;
            timer_device_string = "GPU";
        }
        printf("\tELAPSED TIME: %fms (%s)\n", elapsed_time, timer_device_string);
    }

    void startCpuTimer()
    {
        if (cpu_timer_started) throw std::runtime_error("CPU timer already started");
        cpu_timer_started = true;

        time_start_cpu = std::chrono::high_resolution_clock::now();
    }

    void endCpuTimer()
    {
        time_end_cpu = std::chrono::high_resolution_clock::now();

        if (!cpu_timer_started) throw std::runtime_error("CPU timer not started");

        std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
        prev_elapsed_time_cpu_milliseconds
            = static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

        cpu_timer_started = false;
    }

    void startGpuTimer()
    {
        if (gpu_timer_started) throw std::runtime_error("GPU timer already started");
        gpu_timer_started = true;

        cudaEventRecord(event_start);
    }

    void endGpuTimer()
    {
        cudaEventRecord(event_end);
        cudaEventSynchronize(event_end);

        if (!gpu_timer_started) throw std::runtime_error("GPU timer not started");

        cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
        gpu_timer_started = false;
    }

    float getCpuElapsedTimeForPreviousOperation() const
    {
        return prev_elapsed_time_cpu_milliseconds;
    }

    float getGpuElapsedTimeForPreviousOperation() const
    {
        return prev_elapsed_time_gpu_milliseconds;
    }

    // remove copy and move functions
    PerformanceTimer(const PerformanceTimer&) = delete;
    PerformanceTimer(PerformanceTimer&&) = delete;
    PerformanceTimer& operator=(const PerformanceTimer&) = delete;
    PerformanceTimer& operator=(PerformanceTimer&&) = delete;

private:
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_end = nullptr;

    using time_point_t = std::chrono::high_resolution_clock::time_point;
    time_point_t time_start_cpu;
    time_point_t time_end_cpu;

    float prev_elapsed_time_cpu_milliseconds = 0.f;
    float prev_elapsed_time_gpu_milliseconds = 0.f;
};
}  // namespace Common
}  // namespace StreamCompaction
