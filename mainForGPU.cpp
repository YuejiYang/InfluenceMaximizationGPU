#include <iostream>
<<<<<<< HEAD
#include "GraphStruct.h"
#include "GPUMemManager.h"
#include "GPUcBFS.h"

int main() {
    //read files
    GraphStruct graphStruct = GraphStruct();
    graphStruct.readEdgeList("../data/edges.csv");
    graphStruct.readNodes("../data/nodes.csv");
    graphStruct.writeObjToFile("../data/graph.dat");

    graphStruct.readObjFromFile("../data/graph.dat");

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
=======
#include <fstream>
#include "GraphStruct.hpp"

using namespace std;

int main() {
	GraphStruct<uint32_t> gs, gs2;
	gs.readFromFile("soc-Epinions1.txt");

	ofstream of("out.txt");
	of << gs;

	gs.save("out.dat");

	gs2.load("out.dat");
	ofstream of2("out2.txt");
	of2 << gs2;

	cout << sizeof(gs2.edges[0]);

>>>>>>> ddff1bd5befb4736e55bee2b59b6f6395cccfc7a
    return 0;
}
