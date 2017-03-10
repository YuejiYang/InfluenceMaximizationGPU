#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <assert.h>

//EdgeType must define:
//EdgeType(NodeType, NodeType) constructor
//EdgeType.first  EdgeType.second
//operator<

template <typename NodeType, typename EdgeType=std::pair<NodeType, NodeType> >
class GraphStruct {
public:
	typedef std::vector<NodeType> NodeVec;
	typedef std::vector<EdgeType> EdgeVec;
	typedef std::vector<NodeType> EdgeIdx;

//	Assume Node range from [0,n) for n nodes
//	NodeVec nodes;
	NodeType N;			//#node
	EdgeVec edges;
	EdgeIdx eidx;	//idx of first edge starting from Node_i   ----> size=#node

	void readFromFile(const std::string &filename){
		std::ifstream infile(filename);
		assert(infile.is_open());

		infile >> *this;
	}

	void save(const std::string &filename){
		std::ofstream outfile(filename, std::ios::trunc);
		assert(outfile.is_open());

		unsigned m=edges.size();
		unsigned l=eidx.size();		//should =N
		outfile.write((char*)&N, sizeof(N));
		outfile.write((char*)&m, sizeof(m));
		outfile.write((char*)&l, sizeof(l));
		outfile.write((char*)&edges[0], sizeof(edges[0])*edges.size() );
		outfile.write((char*)&eidx[0], sizeof(eidx[0])*eidx.size() );
	}


	void load(const std::string &filename){
		std::ifstream infile(filename);
		assert(infile.is_open());

		NodeType n;
		unsigned m;
		unsigned l;
		infile.read((char*)&N, sizeof(n));
		infile.read((char*)&m, sizeof(n));
		infile.read((char*)&l, sizeof(l));

		edges.resize(m);
		eidx.resize(l);

		infile.read((char*)&edges[0], m*sizeof(edges[0]));
		infile.read((char*)&eidx[0], l*sizeof(eidx[0]));
	}

	friend std::ostream& operator<<(std::ostream& os, const GraphStruct& gs)
	{
		for(auto &e:gs.edges){
			os << e.first << "," << e.second << "\n";
		}
	}
	friend std::istream& operator>>(std::istream& is, GraphStruct& gs)
	{
		NodeType arg0, arg2;
		std::string arg1;

		gs.N=0;
		while(is >> arg0 >> arg1 >> arg2){
			gs.edges.push_back(EdgeType(arg0, arg2));
			gs.N = std::max(gs.N, std::max(arg0, arg2) );
		}
		gs.N++;	//start from 0

		sort(gs.edges.begin(), gs.edges.end());

		gs.eidx.resize(gs.N);
		int curEdgeIdx = 0;
		for(int i=0;i<gs.N;i++){
			while(gs.edges[curEdgeIdx].first<i && curEdgeIdx<gs.edges.size()){
				curEdgeIdx++;
			}
			gs.eidx[i] = curEdgeIdx;
		}
	}
private:
};
