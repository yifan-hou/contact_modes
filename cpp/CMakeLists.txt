cmake_minimum_required(VERSION 3.2)
project(contact_modes)

set(contact_modes_cmake_dir "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${contact_modes_cmake_dir} ${CMAKE_MODULE_PATH})

set(LINUX UNIX AND NOT APPLE)
if(LINUX)

  # GCC configuration
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")

    # set(GCC_CXX_FLAGS "-std=c++17 -fPIC -march=native -fno-omit-frame-pointer") #-m64
    set(GCC_CXX_FLAGS "-std=c++17") #-m64

    # For compatibility with TCMALLOC from gperftools.
    set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -fno-builtin-malloc")
    set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -fno-builtin-calloc")
    set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -fno-builtin-realloc")
    set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -fno-builtin-free")
    set(CMAKE_SHARED_LINKER_FLAGS "-ltcmalloc_minimal")

    # Debug configuration
    if(BUILD_DEBUG)
        set(CMAKE_BUILD_TYPE Debug)
        set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -g -O2 -fno-omit-frame-pointer")
    else(BUILD_DEBUG)
        set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -g -fno-omit-frame-pointer")
    endif(BUILD_DEBUG)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_CXX_FLAGS}")

    # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
    
  endif()

endif()


find_package(Eigen3 3.3 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Gperftools REQUIRED)
# find_package(PythonLibs 3.6 REQUIRED)

include_directories(
    $(CONTACT_MODES_SOURCE_DIR)/src
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}
)

link_directories(
)

add_subdirectory(src/contact_modes)

option(PACKAGE_TESTS "Build tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# add_subdirectory(apps)
add_subdirectory(bench)