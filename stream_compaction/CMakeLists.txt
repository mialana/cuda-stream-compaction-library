cmake_minimum_required(VERSION 3.24)
project(stream_compaction LANGUAGES CUDA CXX)

set(stream_compaction_VERSION 1.0.0)

set(CURR_HEADERS
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "thrust.h"
    "radix.h"
    "shared.h"
    )

set(CURR_SOURCES
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "thrust.cu"
    "radix.cu"
    "shared.cu"
)

add_library(${PROJECT_NAME} SHARED ${CURR_SOURCES} ${CURR_HEADERS})

target_include_directories(${PROJECT_NAME} PUBLIC 
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)

target_compile_options(${PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;--extended-lambda>")

target_compile_options(${PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;--extended-lambda>")

# packaging

include(CMakePackageConfigHelpers)

set(CMAKE_PKG_FILES_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE INTERNAL "")
set(CMAKE_PKG_FILES_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/cmake" CACHE INTERNAL "")

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets)

# targets file
# install tree usage
install(EXPORT ${PROJECT_NAME}Targets
        FILE "${PROJECT_NAME}Targets.cmake"
        DESTINATION ${CMAKE_PKG_FILES_INSTALL_DIR}
        NAMESPACE ${PROJECT_NAME}::
)
# build tree usage using `install(EXPORT ...)`
export(EXPORT ${PROJECT_NAME}Targets 
    FILE "${CMAKE_PKG_FILES_BUILD_DIR}/${PROJECT_NAME}Targets.cmake"
    NAMESPACE ${PROJECT_NAME}::
)

# config file (will create during configure time)
configure_package_config_file(${CURR_CMAKE_MODULE_ROOT}/Config.cmake.in
  "${CMAKE_PKG_FILES_BUILD_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_PKG_FILES_INSTALL_DIR}
)

# version file (will create during configure time)
write_basic_package_version_file(
    "${CMAKE_PKG_FILES_BUILD_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${stream_compaction_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# install of config and version files
install(FILES
    "${CMAKE_PKG_FILES_BUILD_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_PKG_FILES_BUILD_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKE_PKG_FILES_INSTALL_DIR}
)