cmake_minimum_required(VERSION 3.24)
project(stream_compaction LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # force compilation as separate objects

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# force consistent dynamic (as opposed to static) runtime
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(stream_compaction)

# allow enable/disable of tests target
option(STREAM_COMPACTION_BUILD_TESTS "Build tests" OFF)
if(PROJECT_IS_TOP_LEVEL)
    set(STREAM_COMPACTION_BUILD_TESTS ON) # default for `ExternalProject`, `FetchContent`, etc will be `OFF`
endif()

# configure tests target
if(STREAM_COMPACTION_BUILD_TESTS)
    set(STREAM_COMPACTION_TESTS_TARGET ${PROJECT_NAME}_tests)
    set(STREAM_COMPACTION_TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/tests)

    set(TESTS_HEADERS
        "${STREAM_COMPACTION_TESTS_ROOT}/testing_helpers.h"
    )
    set(TESTS_SOURCES
        "${STREAM_COMPACTION_TESTS_ROOT}/main.cu"
    )

    add_executable(${STREAM_COMPACTION_TESTS_TARGET} ${TESTS_SOURCES} ${TESTS_HEADERS})

    target_link_libraries(${STREAM_COMPACTION_TESTS_TARGET} PRIVATE stream_compaction::stream_compaction)

    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        # make tests target the startup project in VS
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${STREAM_COMPACTION_TESTS_TARGET})
    endif()

    if(WIN32)
        # DLL must sit next to executable on Windows
        add_custom_command(TARGET ${STREAM_COMPACTION_TESTS_TARGET}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:${PROJECT_NAME}>
                $<TARGET_FILE_DIR:${STREAM_COMPACTION_TESTS_TARGET}>
        )
    endif()
endif()

if(MSVC)
    # needed for proper intellisense when using MSVC IDEs (VS, Rider, etc.)
    set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
        "$(VC_IncludePath)"
        "$(WindowsSDK_IncludePath)"
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        "${CMAKE_SOURCE_DIR}"
)
endif()
