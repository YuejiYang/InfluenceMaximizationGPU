//
// Created by yangyueji on 3/21/17.
//

#include "GraphStruct.h"
using namespace std;

GraphStruct::~GraphStruct() {
    //free all pointers
    free(this->allNodes);
    free(this->edgesList);
    free(this->edgesProb);
    free(this->reverseEdgesList);
}

void GraphStruct::readEdgeList(const char *edgeFile){
    vector<string> contents;
    ifstream ifs(edgeFile);
    if(!ifs.is_open()) {
        cout << "cannot open files" << endl;
        exit(0);
    }
    string line;
    while (getline(ifs, line)) {
        if (line[0] == '#') continue;
        if (line.size() > 0) {
            istringstream iss(line);
            vector<uint32_t> tripe;
            uint32_t temp;
            while (iss >> temp) {
                tripe.push_back(temp);
                if (iss.peek() == ',')
                    iss.ignore();
            }
            if (tripe[0] == tripe[1]) continue;

            contents.push_back(line);
        }
    }
    ifs.close();

    this->edgesListSize = (uint32_t)contents.size();

    this->edgesList = (uint64_t*)malloc(this->edgesListSize * sizeof(uint64_t));
    this->reverseEdgesList = (uint64_t*)malloc(this->edgesListSize * sizeof(uint64_t));
    this->edgesProb = (uint16_t*)malloc(this->edgesListSize * sizeof(uint16_t));
    //construct edges
    for (unsigned int i = 0; i < this->edgesListSize; ++i) {
        istringstream iss(contents[i]);
        vector<uint32_t> tripe;
        uint32_t temp;
        while ( iss >> temp ) {
            tripe.push_back(temp);
            if (iss.peek() == ',')
                iss.ignore();
        }

        //node Id start from 1. Use 0 as a special mark
        if(tripe[0] == tripe[1]) continue;
        this->edgesList[i] = combine2u32(tripe[0], tripe[1]);
        this->reverseEdgesList[i] = combine2u32(tripe[1], tripe[0]);

        //write probability
        if (tripe.size() == 3)
            this->edgesProb[i] = (uint16_t)tripe[2];
        else
            this->edgesProb[i] = 10;
    }
}

void GraphStruct::readNodes(const char *nodeFile) {
    ifstream ifs(nodeFile);
    if(!ifs.is_open()) {
        cout << "cannot open files" << endl;
        exit(0);
    }
    std::vector<string> contents;
    string line;
    while (getline(ifs, line)) {
        if (line[0] == '#') continue;
        if(line.size() > 0)
            contents.push_back(line);
    }
    ifs.close();
    this->nodesSize = (uint32_t)contents.size();
    this->allNodes = (uint32_t*)malloc(this->nodesSize * sizeof(uint32_t));
    //for debug
    //cout << "nodeListSize : " << this->nodesSize << endl;

    for (unsigned int i = 0; i < this->nodesSize; ++i) {
        istringstream iss(contents[i]);
        vector<uint32_t> _pairs;
        uint32_t temp;
        while ( iss >> temp ) {
            _pairs.push_back(temp);
            if (iss.peek() == ',')
                iss.ignore();
        }

        if(_pairs[0] > this->nodesSize) {
            cout << "Error : node Id exceed maximum-possible" << endl;
            exit(0);
        }
    }
}

void GraphStruct::writeObjToFile(const char* filename) {
    ofstream ofs(filename, ios::trunc);
    if(!ofs.is_open()) {
        cout << "cannot open files to write" << endl;
        exit(0);
    }
    ofs.write((char*)&this->nodesSize, sizeof(uint32_t));
    ofs.write((char*)&this->edgesListSize, sizeof(uint32_t));

    ofs.write((char*)this->allNodes, sizeof(uint32_t) * this->nodesSize);
    ofs.write((char*)this->edgesList, sizeof(uint64_t) * this->edgesListSize);
    ofs.write((char*)this->edgesProb, sizeof(uint16_t) * this->edgesListSize);
    ofs.write((char*)this->edgesProb, sizeof(uint64_t) * this->edgesListSize);

    ofs.close();
}
void GraphStruct::readObjFromFile(const char* filename) {
    ifstream ifs(filename);
    if(!ifs.is_open()) {
        cout << "cannot open files to read" << endl;
        exit(0);
    }

    ifs.read((char*)&this->nodesSize, sizeof(uint32_t));
    ifs.read((char*)&this->edgesListSize, sizeof(uint32_t));
    this->allNodes = (uint32_t*)malloc(this->nodesSize * sizeof(uint32_t));
    this->edgesList = (uint64_t*)malloc(sizeof(uint64_t) * this->edgesListSize);
    this->reverseEdgesList = (uint64_t*)malloc(sizeof(uint64_t) * this->edgesListSize);
    this->edgesProb = (uint16_t*)malloc(sizeof(uint16_t) * this->edgesListSize);
    ifs.read((char*)this->allNodes, sizeof(uint32_t) * this->nodesSize);
    ifs.read((char*)this->edgesList, sizeof(uint64_t) * this->edgesListSize);
    ifs.read((char*)this->edgesProb, sizeof(uint16_t) * this->edgesListSize);
    ifs.read((char*)this->reverseEdgesList, sizeof(uint64_t) * this->edgesListSize);
    ifs.close();
}

void GraphStruct::readSamples(const char *filename, std::vector<uint32_t>& init_bfs_nodes) {
    ifstream ifs(filename);
    if(!ifs.is_open()) {
        cout << "cannot open files" << endl;
        exit(0);
    }
    string line;
    while (getline(ifs, line)) {
        if (line[0] == '#') continue;
        init_bfs_nodes.push_back((uint32_t)std::stoi(line));
    }

    ifs.close();
}