cmake_minimum_required(VERSION 3.6)

set(PROJECT_NAME InfluenceMaximizationGPU)

project(${PROJECT_NAME})

include(CheckCXXCompilerFlag)
set(CMAKE_CXX_STANDARD 11)

## COMPILERS
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
        "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warnings "-Wall -Wextra -Wno-unused-parameter") #-Werror
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
endif()
if (NOT CONFIGURED_ONCE)
    set(CMAKE_C_FLAGS "${warnings}"
            CACHE STRING "Flags used by the compiler during all build types." FORCE)
    set(CMAKE_CXX_FLAGS "${warnings}"
            CACHE STRING "Flags used by the compiler during all build types." FORCE)
endif()


## DEPENDENCIES
find_package(CUDA REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)


SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 --expt-extended-lambda -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -lcurand -O3)

set(SOURCE_FILES main.cpp
        GraphStruct.cu GraphStruct.h
        GPUKernels.cu GPUKernels.h
        GPUMemManager.cu GPUMemManager.h
        GPUcBFS.cu GPUcBFS.h max_cover/MaxCover.h max_cover/MaxCover.cu)

cuda_add_executable(InfluenceMaximizationGPU ${SOURCE_FILES})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)