#include <iostream>
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

    return 0;
}
