#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <utility>
#include <algorithm>
//#include <functional>

#ifndef SVM_UTIL_H
#define SVM_UTIL_H

using std::vector;
using std::pair;
using std::string;

typedef vector< pair<int, float> > PairArray;

vector<string>& split(const string& str, const string& delims, vector<string>& ret);
bool cmp(const pair<int, float>& lhs, const pair<int, float>& rhs);
void load_data_set(const string& filename, vector<PairArray>& x, vector<float>& y);
float dot_product(const PairArray& arr1, const PairArray& arr2);

#endif