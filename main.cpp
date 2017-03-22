#include <iostream>
#include "GraphStruct.h"
#include "GPUMemManager.h"
#include "GPUcBFS.h"

int main() {
    //read files
    GraphStruct graphStruct = GraphStruct();
    graphStruct.readEdgeList("../data/edges.csv");
    graphStruct.readNodes("../data/nodes.csv");
    graphStruct.writeObjToFile("../data/graph.dat");

    //graphStruct.readObjFromFile("../data/graph.dat");

    GPUMemManager gpuMemManager = GPUMemManager();
    gpuMemManager.initDeviceMem(graphStruct);
    gpuMemManager.sortEdgesOnDev();
    gpuMemManager.setNodeListOnDev(1024, 256);



    std::vector<uint32_t> init_bfs;
    std::vector<std::vector<uint32_t>> leaves;
    uint32_t *status_array = NULL;

    bool satisfied = false;
    while(!satisfied) {

        GPUcBFS::gpucBFS_expansion(init_bfs, gpuMemManager.nodeSize, gpuMemManager.edgeSize,
                                   gpuMemManager.dev_nodeList_raw,
                                   gpuMemManager.dev_edgeList_raw, gpuMemManager.dev_edgeprob_raw, status_array);

        GPUcBFS::gpucBFS_extract_leaves(init_bfs, leaves, status_array, gpuMemManager.nodeSize);

        break;

    }
    return 0;
}
