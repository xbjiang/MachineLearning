/*
* K Nearest Neighbor
*/

#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
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
	void computeAllDist(vector<double>& testVec);
	string classify(vector<double>& testVec);
	struct cmpByValue {
		bool operator() (pair<int, double>& lhs, pair<int, double>& rhs)
		{
			return lhs.second < rhs.second;
		}
	};
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

double KNN::distEuclid(vector<double>& vec1, vector<double>& vec2)
{
	if (vec1.size() != vec2.size())
	{
		cerr << "Size of the inputs has to be same!" << endl;
		exit(1);
	}
	double sum = 0;
	for (int i = 0; i < vec1.size(); i++)
	{
		sum += pow(vec1[i] - vec2[i], 2);
	}
	return sqrt(sum);
}

void KNN::computeAllDist(vector<double>& testVec)
{
	for (int i = 0; i < col; i++)
	{
		double dist = distEuclid(dataSet[i], testVec);
		distVec.push_back(make_pair(i, dist));
	}
}

string KNN::classify(vector<double>& testVec)
{
	if (testVec.size() != col)
	{
		cerr << "Wrong dimension!" << endl;
		exit(1);
	}
	computeAllDist(testVec);
	sort(distVec.begin(), distVec.end(), cmpByValue());
	map<string, int> labelCnt;
	for (int i = 0; i < K; i++)
	{
		int idx = distVec[i].first;
		labelCnt[labelSet[idx]]++;
	}
	int maxCnt = 0;
	string majorLabel = "";
	map<string, int>::iterator iter = labelCnt.begin();
	while (iter != labelCnt.end())
	{
		if (iter->second > maxCnt)
		{
			maxCnt = iter->second;
			majorLabel = iter->first;
		}
		iter++;
	}
	return majorLabel;
}