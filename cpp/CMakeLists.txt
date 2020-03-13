cmake_minimum_required(VERSION 3.2)
project(contact_modes)

set(viscl_cmake_dir "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${viscl_cmake_dir} ${CMAKE_MODULE_PATH})

set(LINUX UNIX AND NOT APPLE)
if(LINUX)

  # GCC configuration
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")

    set(GCC_CXX_FLAGS "-std=c++17 -fPIC -march=native") #-m64

    # X11 Dependencies
    # if(NOT BUILD_LIBCMU462)
    #   set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXi")
    #   set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXxf86vm")
    #   set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXinerama")
    #   set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXcursor")
    #   set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXfixes")
    #   set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXrandr")
    #   set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXext")
    #   set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXrender")
    #   set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lX11")
    #   set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lpthread")
    #   set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lxcb")
    #   set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXau")
    # endif()

    # Debug configuration
    if(BUILD_DEBUG)
        set(CMAKE_BUILD_TYPE Debug)
        # set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -g")
    else(BUILD_DEBUG)
        set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -O3")
        set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -fopenmp")
    endif(BUILD_DEBUG)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_CXX_FLAGS} ")

    # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
    
  endif()

endif()

find_package(Eigen3 3.3 REQUIRED)
find_package(pybind11 REQUIRED)
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