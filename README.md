# InfluenceMaximizationGPU

To keep data format in line:\
Notes:\
    a. Nodes' id start from 0. And the step between two continuous id is always 1.\
    b. nodes.txt flie just list all node id.\
    c. edges.txt keeps a list of tripes line by line. (startNode, endNode, probability). The direction of edges is the same as original data set. Reverse of edges is done while processing input.\
    d. input for concurrent BFS is a vector of all samples\
    e. output from concurrent BFS is a 2D vector. Each line represents all reached nodes by roots. The BFS is done following reverse edges.
    

To compile:\
    mkdir build\
    cd build\
    cmake ..\
    make
    
