#include <iostream>
#include "GraphStruct.h"
#include "GPUMemManager.h"
#include "GPUcBFS.h"

int main() {
    //read files
    GraphStruct graphStruct = GraphStruct();

    graphStruct.readNodes("../data/nodes.txt");
    graphStruct.readEdgeList("../data/edges_with_prob.txt");
    
    //graphStruct.writeObjToFile("../data/graph.dat");
    //graphStruct.readObjFromFile("../data/graph.dat");

    GPUMemManager gpuMemManager = GPUMemManager();
    gpuMemManager.initDeviceMem(graphStruct);
    gpuMemManager.sortEdgesOnDev();
    gpuMemManager.setNodeListOnDev(1024, 256);

    std::vector<uint32_t> init_bfs;
    std::vector<std::vector<uint32_t>> inter_nodes;


    graphStruct.readSamples("../data/samples.txt", init_bfs);
    //graphStruct.readSamples("../data/testNodes.txt", init_bfs);
    int init_bfs_once = 10000;
    int cycles = (int)(init_bfs.size() - 1) / init_bfs_once + 1;

    std::cout << "\nstart concurrent BFS..." << std::endl;
    unsigned long long st = getTime();
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
        //break;

    }
    unsigned long long et = getTime();

    std::cout << "******Total time = " << getInterval(st, et) << "ms. " << std::endl;

    return 0;
}
