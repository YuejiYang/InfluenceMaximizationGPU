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
#include "globalSettings.h"
#include "GPUKernels.h"



//node id should be continuous and starts from 0

class GraphStruct {
public:
    uint32_t nodesSize; //high 16 bits for outgoing
    uint32_t* allNodes;
    uint32_t* allNodes_reverse;

    uint32_t edgesListSize;
    uint64_t* edgesList;
    uint64_t* reverseEdgesList;

    uint16_t* edgesProb;


public:
    GraphStruct() : nodesSize(0), allNodes(NULL),allNodes_reverse(NULL), edgesListSize(0), edgesList(NULL), reverseEdgesList(NULL), edgesProb(NULL) {}
    GraphStruct(int nodeSize, int edgeSize);
    ~GraphStruct();

    void readEdgeList(const char* edgeFile);
    void readNodes(const char* nodeFile);
    void writeObjToFile(const char* filename);
    void readObjFromFile(const char* filename);
    void cpyBackToHost(uint32_t* nodeList_d, uint64_t* edgeList_d);


    void readSamples(const char* filename, std::vector<uint32_t>& init_bfs_nodes);
    void randSample(const int sample_nmb, std::vector<uint32_t>& init_bfs_nodes, unsigned int rand_init = 0);
};


#endif //INFLUENCEMAXIMIZATIONGPU_GRAPHSTRUCT_H
