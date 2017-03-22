//
// Created by yangyueji on 3/21/17.
//

#ifndef INFLUENCEMAXIMIZATIONGPU_GPUCBFS_H
#define INFLUENCEMAXIMIZATIONGPU_GPUCBFS_H

#include <vector>
#include "GPUKernels.h"
namespace GPUcBFS {
    void gpucBFS_expansion(std::vector<uint32_t> init_bfs_nodes,
                           int nodeSize,
                           int edgeSize,
                           uint32_t *dev_nodeList_raw,
                           uint64_t *dev_edgeList_raw,
                           uint16_t *dev_edgeProb_raw,
                           uint32_t *status_array);

    void gpucBFS_extract_leaves(std::vector<uint32_t> init_bfs_nodes,
                                std::vector<std::vector<uint32_t>> & leaves,
                                uint32_t *status_array,
                                const int nodeSize);

};


#endif //INFLUENCEMAXIMIZATIONGPU_GPUCBFS_H
