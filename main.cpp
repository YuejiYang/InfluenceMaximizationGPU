#define HEAD_INFO
#include "para_estimation/head.h"
#include "para_estimation/sfmt/SFMT.h"
#include "para_estimation/graph.h"

#include <iostream>

#include "globalSettings.h"
#include "GraphStruct.h"
#include "GPUMemManager.h"
#include "GPUcBFS.h"
#include "max_cover/MaxCover.h"


int main() {
    cudaSetDevice(1);
    //read files

    string dataset = "../../data/epinions/";
    string model = "IC";
    double epsilon = 0.5;
    int k=100;
    string graph_file = dataset + "graph_ic.inf";

    unsigned long long start_estimate = getTime();
    TimGraph m(dataset, graph_file);
    m.k=k;
    m.setInfuModel(InfGraph::IC);

    double thelta = m.EstimateOPT(epsilon);

    unsigned long long end_estimate = getTime();
    std::cout << "******Estimation Time = " << getInterval(start_estimate, end_estimate) << "ms." << endl;
    //printf("thelta = %.2f\n", thelta);
 
    int sample_nmb = (int) std::ceil(thelta);



    
    GraphStruct graphStruct = GraphStruct();

    graphStruct.readNodes("../../data/nodes.txt");
    graphStruct.readEdgeList("../../data/edges_with_prob.txt");

    GPUMemManager gpuMemManager = GPUMemManager();
    gpuMemManager.initDeviceMem(graphStruct);
    gpuMemManager.sortEdgesOnDev(GPUMemManager::reverseEdgeProcessing);
    gpuMemManager.setNodeListOnDev(1024, 256, GPUMemManager::reverseEdgeProcessing);

//    gpuMemManager.sortEdgesOnDev(GPUMemManager::normalEdgeProcessing);
//    gpuMemManager.setNodeListOnDev(1024, 256, GPUMemManager::normalEdgeProcessing);
//    graphStruct.cpyBackToHost(gpuMemManager.dev_nodeList_raw, gpuMemManager.dev_edgeList_raw);

    unsigned int maxGrid = 1024,  maxBlock = 512;
    gpuMemManager.init_randomStates(maxGrid, maxBlock);

    std::vector<uint32_t> init_bfs;
    std::vector<std::vector<uint32_t>> inter_nodes;
    //graphStruct.readSamples("../../data/nodes.txt", init_bfs);
    graphStruct.randSample(sample_nmb, init_bfs);

    int init_bfs_once = 512;//76ms
    //int init_bfs_once = 1024;//195ms
    //int init_bfs_once = 256;//31ms
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
                                   gpuMemManager.dev_nodeReverseList_raw,
                                   gpuMemManager.dev_reverseEdgeList_raw,
                                   gpuMemManager.dev_edgeprob_raw,
                                   gpuMemManager.all_states,
                                   maxGrid,
                                   maxBlock);
        //break;

    }
    unsigned long long et = getTime();
    std::cout << "******Total BFS time = " << getInterval(st, et) << "ms. " << std::endl;

    //vec<> init_bfs,   vec<vec<>>inter_nodes



    unsigned long long st2 = getTime();
    auto res = maxCoverGreedy(inter_nodes, init_bfs, TOPK);
    unsigned long long et2 = getTime();
    std::cout << "******100-Max cover = " << getInterval(st2, et2) << "ms. " << std::endl;


    return 0;
}
