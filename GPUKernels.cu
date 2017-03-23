//
// Created by yangyueji on 3/21/17.
//

#include "GPUKernels.h"

__global__
void GPUKernels::setNodeListOnDev(uint32_t nodeListSize,
                                  uint32_t edgeListSize,
                                  uint32_t *nodeList_dev_ptr,
                                  uint64_t *edgeList_dev_ptr,
                                  uint32_t maxJobsPerThread,
                                  uint32_t num_threads_one_more_job) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t startIdxOnEdgeList;
    if (tid < num_threads_one_more_job) {
        startIdxOnEdgeList = tid * (maxJobsPerThread + 1);
    } else {
        startIdxOnEdgeList = num_threads_one_more_job * (maxJobsPerThread + 1)
                             + (tid - num_threads_one_more_job) * maxJobsPerThread;
    }

    for (uint32_t i = 0; (tid < num_threads_one_more_job && i < maxJobsPerThread + 1)
                         || (tid >= num_threads_one_more_job && i < maxJobsPerThread); ++i) {
        if (startIdxOnEdgeList + i >= edgeListSize - 1) {
            uint32_t start_node_last = extractHigh32bits(edgeList_dev_ptr[edgeListSize - 1]);
            for(uint32_t j = start_node_last + 1; j < nodeListSize; ++j) {
                nodeList_dev_ptr[j] = edgeListSize;
            }

            break;
        }


        //node Id minus 1 to node Idx. NodeId starts from 1
        uint32_t start_node_1 = extractHigh32bits(edgeList_dev_ptr[startIdxOnEdgeList + i]);
        uint32_t start_node_2 = extractHigh32bits(edgeList_dev_ptr[startIdxOnEdgeList + i + 1]);

        //must handle the first several nodes explicitly
        if (startIdxOnEdgeList + i == 0) {
            for (uint32_t j = 0; j <= start_node_1; ++j) {
                nodeList_dev_ptr[j] = 0;
            }
        }

        if (start_node_1 != start_node_2) {
            for (uint32_t j = 0; j < start_node_2 - start_node_1; ++j) {
                nodeList_dev_ptr[start_node_1 + 1 + j] = startIdxOnEdgeList + i + 1;
            }
        }
    }
}

__global__
void GPUKernels::cBFS_by_warp(const int nodeNum,
                              const int edgeNum,
                              const int curr_level,
                              const int frontier_num,
                              uint8_t *frontier_bmp_raw,
                              const uint32_t *__restrict__ frontier_array_raw,
                              const int status_array_stride,
                              uint8_t *status_array_raw,
                              const uint32_t *__restrict__ nodeList_raw,
                              const uint64_t *__restrict__ edgeList_raw,
                              const uint16_t *__restrict__ edgeProb_raw) {

    int maxWarpNum = gridDim.x * blockDim.x / 32;
    int total_jobs = frontier_num * status_array_stride;
    int maxJobsPerWarp = total_jobs / maxWarpNum;
    int num_warps_one_more_job = total_jobs % maxWarpNum;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = tid / 32;
    int jobs_start, frontier_start;

    if (warp_id < num_warps_one_more_job) {
        jobs_start = warp_id * (maxJobsPerWarp + 1);
    } else {
        jobs_start = num_warps_one_more_job * (maxJobsPerWarp + 1) + (warp_id - num_warps_one_more_job) * maxJobsPerWarp;
    }

    for (unsigned int i = 0;(warp_id < num_warps_one_more_job && i < maxJobsPerWarp + 1) ||
                            (warp_id >= num_warps_one_more_job && i < maxJobsPerWarp); ++i) {
        int warp_offset = tid % 32;
        frontier_start = (jobs_start + i) / status_array_stride; // which node is being handled
        uint32_t curr_node = (uint32_t)frontier_array_raw[frontier_start];

        int status_start = curr_node * status_array_stride;
        int status_offset = (jobs_start + i) % status_array_stride;
        uint8_t original_status = status_array_raw[status_start + status_offset];

        if(original_status != (uint8_t)curr_level) continue;

        uint32_t edge_start, adj_size;
        edge_start = nodeList_raw[curr_node];

        if(curr_node == nodeNum - 1) {
            adj_size = edgeNum - edge_start;
        } else {
            adj_size = nodeList_raw[curr_node + 1] - edge_start;
        }


        curandState s;
        curand_init(curr_node, (uint64_t)tid, 0, &s);

        for(; warp_offset < adj_size; warp_offset+=32) {
            uint32_t end_node = extractLow32bits(edgeList_raw[edge_start + warp_offset]);
            uint8_t end_node_status = status_array_raw[end_node * status_array_stride + status_offset];
            if(end_node_status != ALL_2F) {
                continue;
            }

            uint16_t prob2extend = edgeProb_raw[edge_start + warp_offset];
            float prob_rand = curand_uniform(&s) * 100.0;

            //printf("%d\n", (int)prob_rand);

            if (prob2extend < (uint16_t)prob_rand) {
                continue;
            }
           // printf("@@@@@@@@@,%d\n", end_node);
            status_array_raw[end_node * status_array_stride + status_offset] = (uint8_t)(curr_level + 1);//lock free
            frontier_bmp_raw[end_node] = 1;
        }

    }
}

__global__
void GPUKernels::cBFS_extract_nodes(const int nodeNum, const int status_stride, const int status_offset,
                                     uint8_t *status_array_raw, uint32_t *nodes_bmp) {

    uint32_t maxThreadNum = gridDim.x * blockDim.x;
    uint32_t maxJobsPerThread = nodeNum / maxThreadNum;
    uint32_t num_threads_one_more_job = nodeNum % maxThreadNum;
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t start_idx_on_status_array;

    if (tid < num_threads_one_more_job) {
        start_idx_on_status_array = tid * (maxJobsPerThread + 1);
    } else {
        start_idx_on_status_array =
                num_threads_one_more_job * (maxJobsPerThread + 1) + (tid - num_threads_one_more_job) * maxJobsPerThread;
    }

    for (unsigned int i = 0;(tid < num_threads_one_more_job && i < maxJobsPerThread + 1) ||
                            (tid >= num_threads_one_more_job && i < maxJobsPerThread); ++i) {
        int nodeId = start_idx_on_status_array + i;
        int status_pos = nodeId * status_stride + status_offset;
        uint8_t original_status = status_array_raw[status_pos];
        if(original_status != ALL_2F && original_status != 0) { //exclude roots
           // printf("********%d\n",nodeId);
            nodes_bmp[nodeId] = 1;
        }
    }
}