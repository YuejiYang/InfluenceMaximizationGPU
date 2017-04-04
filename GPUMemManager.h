//
// Created by yangyueji on 3/21/17.
//

#ifndef INFLUENCEMAXIMIZATIONGPU_GPUMEMMANAGER_H
#define INFLUENCEMAXIMIZATIONGPU_GPUMEMMANAGER_H

#include "GraphStruct.h"
class GPUMemManager {
public:

    typedef enum {
        normalEdgeProcessing,
        reverseEdgeProcessing,
    } processingType;

    uint32_t nodeSize;
    uint32_t *dev_nodeList_raw;
    uint32_t *dev_nodeReverseList_raw;

    uint32_t edgeSize;
    uint64_t *dev_reverseEdgeList_raw;
    uint64_t *dev_edgeList_raw;
    uint16_t *dev_edgeprob_raw;

    curandState* all_states;


public:
    GPUMemManager() : nodeSize(0), edgeSize(0) {}
    ~GPUMemManager();

    void initDeviceMem(GraphStruct & graphStruct);
    void sortEdgesOnDev(processingType ptype = normalEdgeProcessing);
    void setNodeListOnDev(uint32_t grid_dim, uint32_t block_dim, processingType ptype = normalEdgeProcessing);
    void init_randomStates(uint32_t maxGrid, uint32_t maxBlock, uint64_t seed = 0);


};


#endif //INFLUENCEMAXIMIZATIONGPU_GPUMEMMANAGER_H
