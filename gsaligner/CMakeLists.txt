# TODO: Change version to at least 3.11 and include FetchContent for Eigen
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(gsaligner LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -Werror)

# this is required only to compile apps in C++, if you work in Python this can be set to off
# if need to compile C++ apps, when executing CMake run `cmake -DCOMPILE_CPP_APPS=ON ..`
option(COMPILE_CPP_APPS "Set to ON to compile C++ applications" OFF)

# Eigen
# TODO: Detect if Eigen exists in system, oterwhise, download it through FetchContent
find_package(Eigen3 3.3 NO_MODULE)

find_package(OpenCV REQUIRED)
message(STATUS "${PROJECT_NAME}: OpenCV version ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

find_package(GTest REQUIRED)

if(${Eigen3_DIR} STREQUAL "Eigen3_DIR-NOTFOUND")
    message(CHECK_START "Fetching Eigen3")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")

    include(FetchContent)
    FetchContent_Declare(
        Eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
    FetchContent_GetProperties(Eigen)

    if(NOT eigen_POPULATED)
        FetchContent_Populate(Eigen)
        add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    list(POP_BACK CMAKE_MESSAGE_INDENT)
    message(CHECK_PASS "fetched")
endif()

find_file(CUDA_HEADER cuda_runtime.h PATHS /usr/include/ /usr/local/cuda/include/)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include ${CUDA_HEADER}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 20012") # TODO: use float* in .cu and Eigen::Matrix in .cpp

find_package(CUDA REQUIRED)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
set(CUDA_ARCH "${ARCH_FLAGS}" CACHE STRING "Value of the NVCC -arch option." FORCE)
# set(CUDA_ARCH "-gencode;arch=compute_75,code=sm_75")

set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -lineinfo -Wno-dev")
message("CUDA ARCH FLAGS: ${ARCH_FLAGS}")
message("CUDA FLAGS: ${CUDA_NVCC_FLAGS}")
message(STATUS "CUDA_ARCH: \"${CUDA_ARCH}\"")

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Python bindings
find_package(pybind11)

if(NOT DEFINED pybind11_FOUND)
    message(CHECK_START "Fetching pybind11")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.12
    )

    FetchContent_GetProperties(pybind11)

    if(NOT pybind11_POPULATED)
        FetchContent_Populate(pybind11)
        add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    endif()

    message(CHECK_PASS "fetched")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PYTHON_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIR}
)

set(TEST_DATA_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/../test_data/")
add_definitions(-DGSTEST_DATA_FOLDER="${TEST_DATA_FOLDER}")
message("${PROJECT_NAME}|test_data dir = ${TEST_DATA_FOLDER}")

add_subdirectory(src)
add_subdirectory(tests)


