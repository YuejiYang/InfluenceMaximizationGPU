#define HEAD_INFO
#include "para_estimation/head.h"

#include "para_estimation/sfmt/SFMT.h"
#include "para_estimation/graph.h"

int main(int argn, char ** argv)
{
    string dataset = "data/epinions/";
    string model="IC";
    double epsilon=0.5;
    int k=100;
    string graph_file=dataset+"graph_ic.inf";
    
    TimGraph m(dataset, graph_file);
    m.k=k;
    m.setInfuModel(InfGraph::IC);

    double thelta = m.EstimateOPT(epsilon);
    printf("thelta = %.2f\n", thelta);
 }