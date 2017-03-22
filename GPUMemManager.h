//
// Created by yangyueji on 3/21/17.
//

#ifndef INFLUENCEMAXIMIZATIONGPU_GPUMEMMANAGER_H
#define INFLUENCEMAXIMIZATIONGPU_GPUMEMMANAGER_H

#include "GraphStruct.h"
class GPUMemManager {
public:
    uint32_t nodeSize;
    uint32_t *dev_nodeList_raw;

    uint32_t edgeSize;
    uint64_t *dev_edgeList_raw;
    uint16_t *dev_edgeprob_raw;
public:
    GPUMemManager() : nodeSize(0), edgeSize(0) {}
    ~GPUMemManager();

    void initDeviceMem(GraphStruct & graphStruct);
    void sortEdgesOnDev();
    void setNodeListOnDev(uint32_t grid_dim, uint32_t block_dim);



};


#endif //INFLUENCEMAXIMIZATIONGPU_GPUMEMMANAGER_H
