#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <stdexcept>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CUDA_CHECK(msg) check_cuda_error_fn(msg, FILENAME, __LINE__)

#define BLOCK_SIZE 128

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void check_cuda_error_fn(const char* msg, const char* file = nullptr, int line = -1);

inline int divup(int size, int div)
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
inline int ilog2_ceil(int x)
{
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}

namespace stream_compaction::common
{
__global__ void kernel_map_to_boolean(int n, const int* idata, int* out_bools);

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
__global__ void kernel_scatter(int n, const int* bools, const int* indices, const int* idata,
                               int* odata);

__global__ void kernel_inclusive_to_exclusive(int n, int identity, const int* idata, int* odata);

__global__ void kernel_set_device_array_value(int* arr, int index, int value);

enum class eTimerDevice
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
        cudaEventCreate(&_event_start);
        cudaEventCreate(&_event_end);
    }

    ~PerformanceTimer()
    {
        cudaEventDestroy(_event_start);
        cudaEventDestroy(_event_end);
    }

    template<eTimerDevice D>
    void start_timer()
    {
        if constexpr (D == eTimerDevice::CPU)
        {
            if (cpu_timer_started) throw std::runtime_error("CPU timer already started");
            cpu_timer_started = true;

            _time_start_cpu = std::chrono::high_resolution_clock::now();
        }
        else
        {
            if (gpu_timer_started) throw std::runtime_error("GPU timer already started");
            gpu_timer_started = true;

            cudaEventRecord(_event_start);
        }
    }

    template<eTimerDevice D>
    void end_timer()
    {
        if constexpr (D == eTimerDevice::CPU)
        {
            _time_end_cpu = std::chrono::high_resolution_clock::now();

            if (!cpu_timer_started) throw std::runtime_error("CPU timer not started");

            std::chrono::duration<double, std::milli> duro = _time_end_cpu - _time_start_cpu;
            _prev_elapsed_time_cpu_milliseconds
                = static_cast<decltype(_prev_elapsed_time_cpu_milliseconds)>(duro.count());

            cpu_timer_started = false;
        }
        else
        {
            cudaEventRecord(_event_end);
            cudaEventSynchronize(_event_end);

            if (!gpu_timer_started) throw std::runtime_error("GPU timer not started");

            cudaEventElapsedTime(&_prev_elapsed_time_gpu_milliseconds, _event_start, _event_end);
            gpu_timer_started = false;
        }
    }

    template<eTimerDevice D>
    inline void print_elapsed_time_for_previous_operation() const
    {
        float elapsed_time;
        const char* timer_device_string;
        if constexpr (D == eTimerDevice::CPU)
        {
            elapsed_time = _prev_elapsed_time_cpu_milliseconds;
            timer_device_string = "CPU";
        }
        else
        {
            elapsed_time = _prev_elapsed_time_gpu_milliseconds;
            timer_device_string = "GPU";
        }
        printf("\tELAPSED TIME: %fms (%s)\n", elapsed_time, timer_device_string);
    }

    void start_cpu_timer()
    {
        if (cpu_timer_started) throw std::runtime_error("CPU timer already started");
        cpu_timer_started = true;

        _time_start_cpu = std::chrono::high_resolution_clock::now();
    }

    void end_cpu_timer()
    {
        _time_end_cpu = std::chrono::high_resolution_clock::now();

        if (!cpu_timer_started) throw std::runtime_error("CPU timer not started");

        std::chrono::duration<double, std::milli> duro = _time_end_cpu - _time_start_cpu;
        _prev_elapsed_time_cpu_milliseconds
            = static_cast<decltype(_prev_elapsed_time_cpu_milliseconds)>(duro.count());

        cpu_timer_started = false;
    }

    void start_gpu_timer()
    {
        if (gpu_timer_started) throw std::runtime_error("GPU timer already started");
        gpu_timer_started = true;

        cudaEventRecord(_event_start);
    }

    void end_gpu_timer()
    {
        cudaEventRecord(_event_end);
        cudaEventSynchronize(_event_end);

        if (!gpu_timer_started) throw std::runtime_error("GPU timer not started");

        cudaEventElapsedTime(&_prev_elapsed_time_gpu_milliseconds, _event_start, _event_end);
        gpu_timer_started = false;
    }

    [[nodiscard]] float get_cpu_elapsed_time_for_previous_operation() const
    {
        return _prev_elapsed_time_cpu_milliseconds;
    }

    [[nodiscard]] float get_gpu_elapsed_time_for_previous_operation() const
    {
        return _prev_elapsed_time_gpu_milliseconds;
    }

    // remove copy and move functions
    PerformanceTimer(const PerformanceTimer&) = delete;
    PerformanceTimer(PerformanceTimer&&) = delete;
    PerformanceTimer& operator=(const PerformanceTimer&) = delete;
    PerformanceTimer& operator=(PerformanceTimer&&) = delete;

private:
    cudaEvent_t _event_start = nullptr;
    cudaEvent_t _event_end = nullptr;

    using time_point_t = std::chrono::high_resolution_clock::time_point;
    time_point_t _time_start_cpu;
    time_point_t _time_end_cpu;

    float _prev_elapsed_time_cpu_milliseconds = 0.f;
    float _prev_elapsed_time_gpu_milliseconds = 0.f;
};
}  // namespace stream_compaction::common
