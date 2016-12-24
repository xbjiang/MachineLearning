#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <utility>
#include <algorithm>
//#include <functional>

#ifndef UTILS_H
#define UTILS_H

using std::vector;
using std::string;
using std::ifstream;

__declspec(dllexport) vector<string>& split(const string& str, const string& delims, vector<string>& ret);
__declspec(dllexport) int read_from_kvfile(ifstream& is, vector< vector<float> >& X, vector<float>& Y);
__declspec(dllexport) void load_data_set(const string& filename, vector< vector<float> >& X, vector<float>& Y);

#endif