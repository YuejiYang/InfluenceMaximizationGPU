//
// Created by yangyueji on 3/21/17.
//

#ifndef INFLUENCEMAXIMIZATIONGPU_GPUCBFS_H
#define INFLUENCEMAXIMIZATIONGPU_GPUCBFS_H

#include <vector>
#include "GPUKernels.h"
namespace GPUcBFS {
    void gpucBFS_expansion(std::vector<uint32_t> init_bfs_nodes,
                           std::vector<std::vector<uint32_t>> &nodes,
                           int nodeSize,
                           int edgeSize,
                           const uint32_t *dev_nodeList_raw,
                           const uint64_t *dev_edgeList_raw,
                           const uint16_t *dev_edgeProb_raw);


};


#endif //INFLUENCEMAXIMIZATIONGPU_GPUCBFS_H
