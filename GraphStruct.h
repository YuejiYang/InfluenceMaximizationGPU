//
// Created by yangyueji on 3/21/17.
//

#ifndef INFLUENCEMAXIMIZATIONGPU_GRAPHSTRUCT_H
#define INFLUENCEMAXIMIZATIONGPU_GRAPHSTRUCT_H

#include <iostream>
#include <stdlib.h>
#include <cstdint>
#include <vector>
#include <fstream>
#include <vector>
#include <sstream>
#include "GPUKernels.h"


//node id should be continuous and starts from 0

class GraphStruct {
public:
    uint32_t nodesSize; //high 16 bits for outgoing
    uint32_t* allNodes;

    uint32_t edgesListSize;
    uint64_t* edgesList;
    uint64_t* reverseEdgesList;

    uint16_t* edgesProb;


public:
    GraphStruct() : nodesSize(0), allNodes(NULL), edgesListSize(0), edgesList(NULL), edgesProb(NULL) {}

    ~GraphStruct();

    void readEdgeList(const char* edgeFile);
    void readNodes(const char* nodeFile);
    void writeObjToFile(const char* filename);
    void readObjFromFile(const char* filename);

    void readSamples(const char* filename, std::vector<uint32_t>& init_bfs_nodes);

};


#endif //INFLUENCEMAXIMIZATIONGPU_GRAPHSTRUCT_H
