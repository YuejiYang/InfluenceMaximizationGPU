//
// Created by yangyueji on 3/21/17.
//

#include "GPUMemManager.h"

using namespace std;
GPUMemManager::~GPUMemManager() {
    cudaFree(this->dev_nodeList_raw);
    cudaFree(this->dev_edgeList_raw);
    cudaFree(this->dev_edgeprob_raw);
    cudaFree(this->all_states);
}

void GPUMemManager::initDeviceMem(GraphStruct &graphStruct) {
    this->nodeSize = graphStruct.nodesSize;
    cudaMalloc((void**)&this->dev_nodeList_raw, this->nodeSize * sizeof(uint32_t));

    //edgeList
    this->edgeSize = graphStruct.edgesListSize;
    cudaMalloc((void**)&this->dev_edgeList_raw, this->edgeSize * sizeof(uint64_t));
    cudaMemcpy(dev_edgeList_raw, graphStruct.reverseEdgesList, sizeof(uint64_t) * this->edgeSize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&this->dev_edgeprob_raw, this->edgeSize * sizeof(uint16_t));
    cudaMemcpy(this->dev_edgeprob_raw, graphStruct.edgesProb, sizeof(uint16_t) * this->edgeSize, cudaMemcpyHostToDevice);

}

void GPUMemManager::sortEdgesOnDev() {
    unsigned long long startTime_sortEdgeList = getTime();
    thrust::device_ptr<uint64_t> dev_edgeList_ptr = thrust::device_pointer_cast(this->dev_edgeList_raw);
    thrust::device_ptr<uint16_t> dev_edgeprob_ptr = thrust::device_pointer_cast(this->dev_edgeprob_raw);
    thrust::sort_by_key(dev_edgeList_ptr, dev_edgeList_ptr + this->edgeSize, dev_edgeprob_ptr);
    unsigned long long endTime_sortEdgeList = getTime();
    cout << "******Edge List sorting time = " << getInterval(startTime_sortEdgeList, endTime_sortEdgeList) << "ms." << endl;

}

void GPUMemManager::init_randomStates(uint32_t maxGrid, uint32_t maxBlock, uint64_t seed) {
    cudaMalloc((void**)&this->all_states, sizeof(curandState) * maxBlock * maxGrid);
    GPUKernels::setup_random_state<<<dim3(maxGrid), dim3(maxBlock)>>>(this->all_states, seed);

}

void GPUMemManager::setNodeListOnDev(uint32_t grid_dim, uint32_t block_dim) {
    dim3 _grid_dim(grid_dim);
    dim3 _block_dim(block_dim);

    uint32_t maxThreadNum = _grid_dim.x * _block_dim.x;
    uint32_t maxJobsPerThread = this->edgeSize / maxThreadNum;
    uint32_t num_threads_one_more_job = this->edgeSize % maxThreadNum;

    unsigned long long startTime_appendAdjList = getTime();
    GPUKernels::setNodeListOnDev<<<_grid_dim, _block_dim >>>(this->nodeSize,
                                 this->edgeSize,
                                 this->dev_nodeList_raw,
                                 this->dev_edgeList_raw,
                                 maxJobsPerThread,
                                 num_threads_one_more_job);
    cudaDeviceSynchronize();
    unsigned long long endTime_appendAdjList = getTime();

    cout << "******Appending adjacent lists to node lists time = " << getInterval(startTime_appendAdjList, endTime_appendAdjList) << "ms." << endl;
}