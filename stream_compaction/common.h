#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CUDA_CHECK(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                       \
                    cudaGetErrorString(err));                                                      \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                                        \
    do                                                                                             \
    {                                                                                              \
        CUDA_CHECK(cudaGetLastError());                                                            \
    } while (0)

constexpr int kBLOCK_SIZE = 128;

// Check for CUDA errors; print and exit if there was a problem.
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

    template<eTimerDevice Device>
    void start_timer()
    {
        if constexpr (Device == eTimerDevice::CPU)
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

    template<eTimerDevice Device>
    void end_timer()
    {
        if constexpr (Device == eTimerDevice::CPU)
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

    template<eTimerDevice Device>
    inline void flush() const
    {
        float elapsed_time;
        const char* timer_device_string;
        if constexpr (Device == eTimerDevice::CPU)
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

    template<eTimerDevice Device>
    [[nodiscard]] float get_elapsed_time_for_previous_operation() const
    {
        if constexpr (Device == eTimerDevice::CPU)
        {
            return _prev_elapsed_time_cpu_milliseconds;
        }
        else
        {
            return _prev_elapsed_time_gpu_milliseconds;
        }
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
