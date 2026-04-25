find_package(CUDAToolkit REQUIRED)

set(LIB_HEADERS
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "thrust.h"
    "radix.h"
    "shared.h"
)

set(LIB_SOURCES
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "thrust.cu"
    "radix.cu"
    "shared.cu"
)

add_library(${PROJECT_NAME} SHARED ${LIB_SOURCES} ${LIB_HEADERS})

target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>" # one directory up
)

find_package(CCCL REQUIRED CONFIG)
target_link_libraries(${PROJECT_NAME} PRIVATE CCCL::CCCL PUBLIC CUDA::cudart)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "native")

target_compile_options(${PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;--extended-lambda>")

target_compile_options(${PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;--extended-lambda>")

# create a namespaced alias to distinguish what is actually linked against
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})