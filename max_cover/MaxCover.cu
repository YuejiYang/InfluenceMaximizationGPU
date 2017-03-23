#include "MaxCover.h"
#include <map>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <functional>

using namespace std;


struct hash_functor{
	size_t operator()(const pair<unsigned, unsigned> &p) const{
		std::hash<unsigned> hash_fn;
		size_t a = hash_fn(p.first);
		size_t b = hash_fn(p.second);
		return (a<<4)|((a>>4)^b);
	}
};

typedef unordered_map<pair<unsigned, unsigned>, unsigned, hash_functor> NextMap;


void transformSets(vector<vector<unsigned > > &sets, NextMap &nextElem, unordered_map<unsigned, unsigned> &firstElem,
		unordered_map<unsigned, unordered_set<unsigned> > &vecSets)
{
	unordered_map<unsigned, unsigned> curElem;

	for(int i=0;i<sets.size();i++){
		unordered_set<unsigned> si(sets[i].begin(), sets[i].end());
		vecSets.insert(make_pair((unsigned)i, si) );

		for(auto elem:sets[i]){
			if(firstElem.find(elem)!=firstElem.end()){
				firstElem[elem] = i;
				curElem[elem] = i;
			} else{
				//  c(e),e|->i  && c(e)=i
				nextElem[make_pair(curElem[elem], elem)] = i;
				curElem[elem] = i;
			}
		}
	}
}


std::vector<unsigned> maxCoverGreedy(std::vector<std::vector<unsigned> > &sets, std::vector<unsigned> &roots, int k)
{

	NextMap nextElem;
	unordered_map<unsigned, unsigned> firstElem;
	unordered_map<unsigned, unordered_set<unsigned> > vecSets;

	transformSets(sets, nextElem, firstElem, vecSets);


	vector<unsigned> results;

	for(int i=0;i<k;i++){
		pair<int, int> max_set(-1, -1);
		for(int k=i;k<vecSets.size();k++){
			max_set = max(max_set, pair<int, int>(vecSets[k].size(), k));
		}

		for(unsigned elem:vecSets[max_set.second]){
			unsigned cur = firstElem[elem];
			while(true){
				//delete elem from cur
				vecSets[cur].erase(elem);

				auto kp = make_pair(cur, elem);
				auto iter = nextElem.find(kp);
				if(iter==nextElem.end()){
					break;
				}
				cur = iter->second;
			}
		}

		results.push_back(roots[max_set.second]);
		vecSets.erase(max_set.second);
	}

	return results;
}
