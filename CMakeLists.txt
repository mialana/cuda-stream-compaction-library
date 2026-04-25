cmake_minimum_required(VERSION 3.24)
project(cuda_stream_compaction LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CURR_CMAKE_MODULE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/cmake" CACHE INTERNAL "") # make a variable for cmake dir

set(TESTS_TARGET_NAME ${PROJECT_NAME}_tests)

set(CURR_HEADERS
    "tests/testing_helpers.h"
)

set(CURR_SOURCES
    "tests/main.cu"
)

add_executable(${TESTS_TARGET_NAME} ${CURR_SOURCES} ${CURR_HEADERS})

add_subdirectory(stream_compaction)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_PKG_FILES_BUILD_DIR}")
find_package(stream_compaction CONFIG)

target_link_libraries(${TESTS_TARGET_NAME} stream_compaction::stream_compaction)

if(CMAKE_GENERATOR MATCHES "Visual Studio")
    # make tests target the startup project in VS
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TESTS_TARGET_NAME})
endif()

