//
// Created by yangyueji on 3/21/17.
//

#include "GPUcBFS.h"

using namespace std;


void GPUcBFS::gpucBFS_expansion(std::vector<uint32_t> init_bfs_nodes,
                                std::vector<std::vector<uint32_t>> & nodes,
                                int nodeSize,
                                int edgeSize,
                                const uint32_t *dev_nodeList_raw,
                                const uint64_t *dev_edgeList_raw,
                                const uint16_t *dev_edgeProb_raw) {
    uint32_t max_level = 20;
    uint8_t *frontier_bmp;
    cudaMalloc((void **) &frontier_bmp, sizeof(uint8_t) * nodeSize);
    thrust::device_ptr<uint8_t> frontier_bmp_ptr = thrust::device_pointer_cast(frontier_bmp);
    thrust::fill(frontier_bmp_ptr, frontier_bmp_ptr + nodeSize, 0u);

    uint8_t *status_array;
    int status_array_stride = (int) init_bfs_nodes.size();
    uint64_t status_array_size = (uint64_t) status_array_stride * nodeSize;
    status_array = (uint8_t *) malloc(sizeof(uint8_t) * nodeSize);
    uint8_t *status_array_d;
    cudaMalloc((void **) &status_array_d, sizeof(uint8_t) * status_array_size);
    std::fill(status_array, status_array + status_array_size, ALL_F);
    for (unsigned int i = 0; i < init_bfs_nodes.size(); ++i) {
        uint32_t nodeId = init_bfs_nodes[i];
        uint32_t status_start = nodeId * status_array_stride;
        status_array[status_start + i] = 0;//parentID = self's
        frontier_bmp_ptr[nodeId] = 1;
    }
    cudaMemcpy(status_array_d, status_array, sizeof(uint8_t) * status_array_size, cudaMemcpyHostToDevice);

    thrust::device_ptr<uint8_t> status_ptr = thrust::device_pointer_cast(status_array_d);


    int frontier_num = (int) init_bfs_nodes.size();
    uint32_t *curr_frontier_raw;
    cudaMalloc((void **) &curr_frontier_raw, sizeof(uint32_t) * nodeSize);
    thrust::device_ptr<uint32_t> curr_frontier_ptr = thrust::device_pointer_cast(curr_frontier_raw);
    thrust::counting_iterator<uint32_t> _first_id(0);
    thrust::counting_iterator<uint32_t> _last_id = _first_id + nodeSize;

    double prepare_time = 0;
    double kernel_time = 0;

    for (unsigned int level = 0; level < max_level; ++level) {
        unsigned long long prepare1 = getTime();
        thrust::copy(_first_id, _last_id, curr_frontier_ptr);
        thrust::sort_by_key(frontier_bmp_ptr, frontier_bmp_ptr + nodeSize, curr_frontier_ptr,
                            thrust::greater<uint32_t>());
        frontier_num = thrust::reduce(frontier_bmp_ptr, frontier_bmp_ptr + nodeSize, (uint32_t) 0);//this third parameter cannot be (uint8_t)0
        cout << frontier_num << endl;
        thrust::fill(frontier_bmp_ptr, frontier_bmp_ptr + nodeSize, 0u);
        unsigned long long prepare2 = getTime();
        prepare_time += getInterval(prepare1, prepare2);

        if (frontier_num <= 0) {
            cout << "No more frontiers!" << endl;
            break;
        }

        unsigned long long kernel1 = getTime();
        GPUKernels::cBFS_by_warp<<<dim3(1024), dim3(256)>>>(nodeSize,
                                 edgeSize,
                                 level,
                                 frontier_num,
                                 frontier_bmp,
                                 curr_frontier_raw,
                                 status_array_stride,
                                 status_array_d,
                                 dev_nodeList_raw,
                                 dev_edgeList_raw,
                                 dev_edgeProb_raw);

        cudaDeviceSynchronize();
        unsigned long long kernel2 = getTime();
        kernel_time += getInterval(kernel1, kernel2);

        for(unsigned int j = 0; j < nodeSize; ++j ) {
            cout << j << " : \n----";
            for(unsigned int k = 0; k < init_bfs_nodes.size(); ++k) {
                cout << (int)status_ptr[j*status_array_stride + k] << " , ";
            }
            cout << endl;
        }


    }

    uint8_t* leaves_bmp;
    cudaMalloc((void**)&leaves_bmp, sizeof(uint8_t) * nodeSize);
    thrust::device_ptr<uint8_t> leaves_bmp_ptr = thrust::device_pointer_cast(leaves_bmp);
    thrust::device_vector<int> node_id((uint64_t)nodeSize);

    for(unsigned int i = 0; i < init_bfs_nodes.size(); ++i) {
        thrust::fill(leaves_bmp_ptr, leaves_bmp_ptr + nodeSize, 0u);
        GPUKernels::cBFS_extract_nodes<<<dim3(1024), dim3(256)>>>(nodeSize,
                status_array_stride,
                i,
                status_array_d,
                leaves_bmp);

        cudaDeviceSynchronize();
        uint32_t inter_nodes_size = thrust::reduce(leaves_bmp_ptr, leaves_bmp_ptr + nodeSize, (uint32_t)0);
        thrust::copy(_first_id, _last_id, node_id.begin());
        thrust::sort_by_key(leaves_bmp_ptr, leaves_bmp_ptr + nodeSize, node_id.begin(), thrust::greater<uint32_t>());
        std::vector<uint32_t> leavesOne(inter_nodes_size);
        thrust::copy(node_id.begin(), node_id.begin() + inter_nodes_size, leavesOne.begin());
        nodes.push_back(leavesOne);

    }

    for (int l = 0; l < nodes.size(); ++l) {
        cout << l << " : \n----";
        for (int m = 0; m < nodes[l].size(); ++m) {
            cout << nodes[l][m] << " , ";
        }
        cout << endl;
    }


    cudaFree(leaves_bmp);
    cudaFree(curr_frontier_raw);
    cudaFree(frontier_bmp);
    cudaFree(status_array_d);
}
