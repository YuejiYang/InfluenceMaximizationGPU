#include <iostream>
#include "GraphStruct.h"
#include "GPUMemManager.h"
#include "GPUcBFS.h"

int main() {
    //read files
    GraphStruct graphStruct = GraphStruct();
    graphStruct.readEdgeList("../data/testEdges.txt");
    graphStruct.readNodes("../data/testNodes.txt");
    graphStruct.writeObjToFile("../data/testGraph.dat");

    //graphStruct.readObjFromFile("../data/graph.dat");

    GPUMemManager gpuMemManager = GPUMemManager();
    gpuMemManager.initDeviceMem(graphStruct);
    gpuMemManager.sortEdgesOnDev();
    gpuMemManager.setNodeListOnDev(1024, 256);

    std::vector<uint32_t> init_bfs;
    std::vector<std::vector<uint32_t>> inter_nodes;


    graphStruct.readSamples("../data/samples.txt", init_bfs);

    int init_bfs_once = 100;
    int cycles = (int)(init_bfs.size() - 1) / init_bfs_once + 1;

    init_bfs.push_back(0);
    init_bfs.push_back(8);

    for (int l = 0; l < cycles; ++l) {
        int total_num_init_bfs = init_bfs_once;
        int init_bfs_start = init_bfs_once * l;
        if( l == cycles - 1) {
            total_num_init_bfs = (int)init_bfs.size() - init_bfs_start;
        }
        std::vector<uint32_t> temp_init_bfs((uint64_t)total_num_init_bfs);
        std::copy(init_bfs.begin() + init_bfs_start, init_bfs.begin() + init_bfs_start + total_num_init_bfs, temp_init_bfs.begin());
        GPUcBFS::gpucBFS_expansion(temp_init_bfs,
                                   inter_nodes,
                                   gpuMemManager.nodeSize,
                                   gpuMemManager.edgeSize,
                                   gpuMemManager.dev_nodeList_raw,
                                   gpuMemManager.dev_edgeList_raw,
                                   gpuMemManager.dev_edgeprob_raw);

    }

    return 0;
}
