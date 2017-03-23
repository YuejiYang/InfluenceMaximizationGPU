#include "MaxCover.h"
#include <iostream>
#include <vector>

using namespace std;


int main()
{
//	10 5
//	5 7 8 9 10
//	5 7
//	1 3 6 10
//	4 6 7 9 10
//	1 2 8 10
	int inp[] = {10,5,5,7,8,9,10,5,7,1,3,6,10,4,6,7,9,10,1,2,8,10};
	int inpsize[] = {2,5,2,4,5,4};
	int inpbeg[] = {0,2,7,9,13,18,22};

	vector<vector<unsigned> > sets;
	vector<unsigned> root;

	for(int i=0;i<6;i++){
		vector<unsigned> s(inp+inpbeg[i], inp+inpbeg[i+1]);
		sets.push_back(s);
		root.push_back(i);
	}

	cout << "computing" << endl;

	auto res = maxCoverGreedy(sets, root, 3);

	for(auto r:res){
		cout << r << endl;
	}
	return 0;
}
