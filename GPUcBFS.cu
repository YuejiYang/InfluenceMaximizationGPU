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
                                const uint16_t *dev_edgeProb_raw,
                                curandState* all_state,
                                unsigned int maxGrid, unsigned int maxBlock) {
    uint32_t max_level = MAX_DEPTH;
    uint8_t *frontier_bmp;
    cudaMalloc((void **) &frontier_bmp, sizeof(uint8_t) * nodeSize);
    thrust::device_ptr<uint8_t> frontier_bmp_ptr = thrust::device_pointer_cast(frontier_bmp);
    thrust::fill(frontier_bmp_ptr, frontier_bmp_ptr + nodeSize, 0u);

    uint8_t *status_array;
    int status_array_stride = (int) init_bfs_nodes.size();
    uint64_t status_array_size = (uint64_t) status_array_stride * nodeSize;
    status_array = (uint8_t *) malloc(sizeof(uint8_t) * status_array_size);
    uint8_t *status_array_d;
    cudaMalloc((void **) &status_array_d, sizeof(uint8_t) * status_array_size);
    std::fill(status_array, status_array + status_array_size, ALL_2F);
    for (unsigned int i = 0; i < init_bfs_nodes.size(); ++i) {
        uint32_t nodeId = init_bfs_nodes[i];
        uint32_t status_start = nodeId * status_array_stride;
        status_array[status_start + i] = 0;//parentID = self's
        frontier_bmp_ptr[nodeId] = 1;
    }
    cudaMemcpy(status_array_d, status_array, sizeof(uint8_t) * status_array_size, cudaMemcpyHostToDevice);

//    thrust::device_ptr<uint8_t> status_ptr = thrust::device_pointer_cast(status_array_d);


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
        //cout << frontier_num << endl;
        thrust::fill(frontier_bmp_ptr, frontier_bmp_ptr + nodeSize, 0u);
        unsigned long long prepare2 = getTime();
        prepare_time += getInterval(prepare1, prepare2);

        if (frontier_num <= 0) {
            //cout << "No more frontiers!" << endl;
            break;
        }
//        for(unsigned int j = 0; j < frontier_num; ++j ) {
//            int node_now = curr_frontier_ptr[j];
//            cout << node_now << " : \n----";
//            for(unsigned int k = 0; k < init_bfs_nodes.size(); ++k) {
//                cout << (int)status_ptr[node_now*status_array_stride + k] << " , ";
//            }
//            cout << endl;
//        }

        unsigned long long kernel1 = getTime();
        GPUKernels::cBFS_by_warp<<<dim3(maxGrid), dim3(maxBlock)>>>(nodeSize,
                                 edgeSize,
                                 level,
                                 frontier_num,
                                 frontier_bmp,
                                 curr_frontier_raw,
                                 status_array_stride,
                                 status_array_d,
                                 dev_nodeList_raw,
                                 dev_edgeList_raw,
                                 dev_edgeProb_raw,
                                 all_state);

        cudaDeviceSynchronize();
        unsigned long long kernel2 = getTime();
        kernel_time += getInterval(kernel1, kernel2);

//        for(unsigned int j = 0; j < frontier_num; ++j ) {
//            int node_now = curr_frontier_ptr[j];
//            cout << node_now << " : \n----";
//            for(unsigned int k = 0; k < init_bfs_nodes.size(); ++k) {
//                cout << (int)status_ptr[node_now*status_array_stride + k] << " , ";
//            }
//            cout << endl;
//        }

    }

    if(frontier_num > 0) {
        cout << "reach max level!" << endl;
    }
    //cout << kernel_time << " : ";

//    uint32_t* leaves_bmp;
//    cudaMalloc((void**)&leaves_bmp, sizeof(uint32_t) * nodeSize);
//    thrust::device_ptr<uint32_t> leaves_bmp_ptr = thrust::device_pointer_cast(leaves_bmp);
//    thrust::device_vector<int> node_id((uint64_t)nodeSize);

    unsigned long long agg1 = getTime();

//    double gpu_extract_time = 0.0;
//    double mem_time = 0.0;
//    for(unsigned int i = 0; i < init_bfs_nodes.size(); ++i) {
//
//        unsigned long long extract1 = getTime();
//
//        thrust::fill(leaves_bmp_ptr, leaves_bmp_ptr + nodeSize, 0u);
//        GPUKernels::cBFS_extract_nodes<<<dim3(maxGrid), dim3(maxBlock)>>>(nodeSize,
//                status_array_stride,
//                i,
//                status_array_d,
//                leaves_bmp);
//
//        cudaDeviceSynchronize();
//        unsigned long long extract2 = getTime();
//        uint32_t inter_nodes_size = thrust::reduce(leaves_bmp_ptr, leaves_bmp_ptr + nodeSize, (uint32_t)0);
//        unsigned long long extract3 = getTime();
//        thrust::copy(_first_id, _last_id, node_id.begin());
//
//        thrust::sort_by_key(leaves_bmp_ptr, leaves_bmp_ptr + nodeSize, node_id.begin(), thrust::greater<uint32_t>());
//
//
//        //gpu_extract_time += getInterval(extract1, extract2);
//        gpu_extract_time += getInterval(extract2, extract3);
//
//        unsigned long long extract11 = getTime();
//        std::vector<uint32_t> leavesOne(inter_nodes_size);
//        thrust::copy(node_id.begin(), node_id.begin() + inter_nodes_size, leavesOne.begin());
//        nodes.push_back(leavesOne);
//
//        unsigned long long extract22 = getTime();
//        mem_time += getInterval(extract11, extract22);
//
//    }
//    unsigned long long agg2 = getTime();
//    cout << getInterval(agg1, agg2) << endl;
//
//    cout << "extractTime = " << gpu_extract_time << endl;
//    cout << "memcpy time = " << mem_time << endl;
    //    for (int l = 0; l < nodes.size(); ++l) {
//        cout << l << " : \n----";
//        for (int m = 0; m < nodes[l].size(); ++m) {
//            cout << nodes[l][m] << " , ";
//        }
//        cout << endl;
//    }


    int init_bfs_num = (int)init_bfs_nodes.size();
    //uint32_t* bfs_index_h = (uint32_t*)malloc(sizeof(uint32_t) * init_bfs_num);
    uint32_t* bfs_index_d;
    cudaMalloc((void**)&bfs_index_d, sizeof(uint32_t) * init_bfs_num);
    thrust::device_ptr<uint32_t> bfs_index_ptr = thrust::device_pointer_cast(bfs_index_d);
    thrust::fill(bfs_index_ptr, bfs_index_ptr + init_bfs_num, 0);
    uint64_t cs1 = getTime();
    GPUKernels::calculate_space_nodes<<<dim3(maxGrid), dim3(maxBlock)>>>(nodeSize, status_array_stride, init_bfs_num, status_array_d, bfs_index_d);
    //cudaMemcpy(bfs_index_h, bfs_index_d, sizeof(uint32_t) * init_bfs_num, cudaMemcpyDeviceToHost);//no need
    thrust::inclusive_scan(bfs_index_ptr, bfs_index_ptr + init_bfs_num, bfs_index_ptr);

    //cout << total_nodes_size << endl;
    //write out the results
    uint32_t total_inter_nodes_size = bfs_index_ptr[init_bfs_num - 1];
    uint32_t* interNodes_container_d;
    uint32_t* bfs_index_rep_d;
    cudaMalloc((void**)&bfs_index_rep_d, sizeof(uint32_t) * init_bfs_num);
    thrust::device_ptr<uint32_t> bfs_index_rep_ptr = thrust::device_pointer_cast(bfs_index_rep_d);
    thrust::fill(bfs_index_rep_ptr, bfs_index_rep_ptr + init_bfs_num, 0);
    cudaMalloc((void**)&interNodes_container_d, sizeof(uint32_t) * total_inter_nodes_size);
    thrust::device_ptr<uint32_t> interNodes_ptr = thrust::device_pointer_cast(interNodes_container_d);
    GPUKernels::extract_nodes<<<dim3(maxGrid), dim3(maxBlock)>>>(interNodes_container_d,
                              nodeSize,
                              status_array_stride,
                              init_bfs_num,
                              status_array_d,
                              bfs_index_d,
                              bfs_index_rep_d);




    for (unsigned int l = 0; l < (unsigned int)init_bfs_num; ++l) {
        uint32_t write_start, write_sz;
        if(l==0) {
            write_start = 0;
            write_sz = bfs_index_ptr[0];
        }
        else {
            write_start = bfs_index_ptr[l - 1];
            write_sz = bfs_index_ptr[l] - write_start;
        }
        std::vector<uint32_t> oneLine(write_sz);
        thrust::copy(interNodes_ptr + write_start, interNodes_ptr + write_start + write_sz, oneLine.begin());
//        if(l==19) {
//            cout << write_start << ";" <<write_sz << endl;
//            for (int m = 0; m < write_sz; ++m) {
//                cout << interNodes_ptr[write_start + m] << endl;
//
//            }
//        }

        nodes.push_back(oneLine);



    }

    uint64_t cs2 = getTime();
    //cout << getInterval(cs1, cs2) << endl;





   // free(bfs_index_h);
    cudaFree(bfs_index_d);
   //cudaFree(leaves_bmp);
    cudaFree(curr_frontier_raw);
    cudaFree(frontier_bmp);
    cudaFree(status_array_d);
    cudaFree(bfs_index_rep_d);
    cudaFree(interNodes_container_d);
}
