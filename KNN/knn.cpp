/*
* K Nearest Neighbor
*/

#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdlib.h>
using namespace std;

class KNN
{
private:
	vector< vector<double> > dataSet;
	vector<string> labelSet;
	vector< pair<int, double> > distVec;
	int K;
	int row, col;

public:
	KNN(char* filename, int k, int dim);
	double distEuclid(vector<double>& vec1, vector<double>& vec2);
	void loadDataSet(char* filename, int dim);
};

KNN::KNN(char* filename, int k, int dim) : K(K), col(dim)
{
	loadDataSet(filename, dim);
}

void KNN::loadDataSet(char* filename, int dim)
{
	ifstream fin(filename);
	if (!fin)
	{
		cerr << "opening file" << filename << "failed!" << endl;
		exit(1);
	}
	string buffer = "";
	while (getline(fin, buffer)) // has to #include <sstream> ? why? do some research!
	{
		istringstream iss(buffer);
		double tmp;
		vector<double> itemLine;
		for (int i = 0; i < dim; i++)
		{
			iss >> tmp;
			itemLine.push_back(tmp);
		}
		dataSet.push_back(itemLine);
		string label;
		iss >> label;
		labelSet.push_back(label);
	}
	row = labelSet.size();
	fin.close();
}