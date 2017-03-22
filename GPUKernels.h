//
// Created by yangyueji on 3/21/17.
//

#ifndef INFLUENCEMAXIMIZATIONGPU_GPUKERNELS_H
#define INFLUENCEMAXIMIZATIONGPU_GPUKERNELS_H


#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cstdint>
#include <curand_kernel.h>


#define ALL_01 0x55555555
#define ALL_10 0xAAAAAAAA
#define SCORES_LOW_16 0x0000FFFF
#define HIGH_16 0xFFFF0000
#define ALL_F 0xFFFFFFFF
inline
unsigned long long getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long long ret = (unsigned long long)tv.tv_usec;
    /* Adds the seconds after converting them to microseconds (10^-6) */
    ret += (tv.tv_sec * 1000 * 1000);
    return ret;
};


inline
double getInterval(unsigned long long start, unsigned long long stop) {
    return (double) (stop - start) / 1000.0;
};



__host__ __device__
__inline__
uint32_t extractHigh32bits(uint64_t num) {
    return (uint32_t) ((num >> 32) & ALL_F);
};

__host__ __device__
__inline__
uint32_t extractLow32bits(uint64_t num) {
    return (uint32_t) (num & ALL_F);
};


__host__ __device__
__inline__
uint64_t combine2u32(uint32_t high, uint32_t low) {
    uint64_t res = 0;
    res = res | high;
    res = res << 32;
    res = res | low;
    return res;

};


namespace GPUKernels {

    __global__ void setNodeListOnDev(uint32_t edgeListSize,
                                     uint32_t *nodeList_dev_ptr,
                                     uint64_t *edgeList_dev_ptr,
                                     uint32_t maxJobsPerThread,
                                     uint32_t num_threads_one_more_job);

    __global__ void cBFS_by_warp(const int nodeNum,
                                 const int edgeNum,
                                 const int curr_level,
                                 const int frontier_num,
                                 uint8_t *frontier_bmp_raw,
                                 uint32_t *frontier_array_raw,
                                 const int status_array_stride,
                                 uint32_t *status_array_raw,
                                 const uint32_t *__restrict__ nodeList_raw,
                                 const uint64_t *__restrict__ edgeList_raw,
                                 const uint16_t *__restrict__ edgeProb_raw);

    __global__ void cBFS_extract_leaves(const int nodeNum,
                                        const int status_stride,
                                        const int status_offset,
                                        uint32_t *status_array_raw,
                                        uint8_t* leaves_bmp);
};


#endif //INFLUENCEMAXIMIZATIONGPU_GPUKERNELS_H