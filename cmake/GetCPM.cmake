if(CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_PATH "${CPM_SOURCE_CACHE}/cmake/CPM.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_PATH "$ENV{CPM_SOURCE_CACHE}/cmake/CPM.cmake")
else()
    set(CPM_DOWNLOAD_PATH "${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_PATH}))
  message(STATUS "Downloading `CPM.cmake` to '${CPM_DOWNLOAD_PATH}'")
  file(DOWNLOAD
       https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake
       ${CPM_DOWNLOAD_PATH}
  )
endif()

include(${CPM_DOWNLOAD_PATH})