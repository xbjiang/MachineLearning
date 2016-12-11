/*
* K Nearest Neighbor
*/

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <utility>
using namespace std;

class KNN
{
private:
	vector< vector<double> > trainSet;
    vector< vector<double> > testSet;
    double ratio; // size of trainSet to testSet 
	vector<string> trainLabel;
    vector<string> testLabel;
	vector< pair<int, double> > distVec;
	int K;
    int dim; // dimension of the dataSet

public:
	KNN(const char* filename, int k, double r, int d);
	double distEuclid(vector<double>& vec1, vector<double>& vec2);
	void loadDataSet(const char* filename);
	void computeAllDist(vector<double>& testVec);
	string classify(vector<double>& testVec);
    double computeAccuracy(bool verbose);
    vector<string>& split(const string& str, const char delim, vector<string>& ret);
    vector<string>& split(const string& str, const string& delims, vector<string>& ret);
	struct cmpByValue {
		bool operator() (pair<int, double>& lhs, pair<int, double>& rhs)
		{
			return lhs.second < rhs.second;
		}
	};
};

KNN::KNN(const char* filename, int k, double r, int d) : K(k), ratio(r), dim(d)
{
	loadDataSet(filename);
}

void KNN::loadDataSet(const char* filename)
{
	ifstream fin(filename);
	if (!fin)
	{
		cerr << "opening file" << filename << "failed!" << endl;
		exit(1);
	}
	string itemLine = "";
    while (getline(fin, itemLine)) // has to #include <sstream> ? why? do some research!
    {
        vector<string> items;
        double prob = double(rand()) / (double)RAND_MAX;
        split(itemLine, ',', items);
        vector<double> dataRow;
        for (int i = 0; i < dim; i++)
            dataRow.push_back(atof(items[i].c_str())); // both g++ and vs seem to have a bug about stof, use atof(str.c_str()) instead
        if (prob > ratio)
        {
            trainSet.push_back(dataRow);
            trainLabel.push_back(items[dim]);
        }
        else
        {
            testSet.push_back(dataRow);
            testLabel.push_back(items[dim]);
        }
    }
    fin.close();
}

/*
* split a string, with one sigle delimiter
*/
vector<string>& KNN::split(const string& str, const char delim, vector<string>& ret)
{
    istringstream iss(str);
    string item;
    while (getline(iss, item, delim))
    {
        if (item.empty()) continue;
        ret.push_back(item);
    }
    return ret;
}

/*
* split a string, with multiple delimiters
*/
vector<string>& KNN::split(const string& str, const string& delims, vector<string>& ret)
{
    string::size_type pos, prev = 0;
    while ((pos = str.find_first_of(delims, prev)) != string::npos)
    {
        if (pos > prev)
        {
            ret.emplace_back(str, prev, pos - prev);
        }
        prev = pos + 1;
    }
    if (prev < str.size()) ret.emplace_back(str, prev, str.size() - prev);
    return ret;
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
    distVec.clear();
	for (int i = 0; i < trainSet.size(); i++)
	{
		double dist = distEuclid(trainSet[i], testVec);
		distVec.push_back(make_pair(i, dist));
	}
}

string KNN::classify(vector<double>& testVec)
{
	if (testVec.size() != dim)
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
		labelCnt[trainLabel[idx]]++;
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

double KNN::computeAccuracy(bool verbose=true)
{
    int cnt = 0;
    int n = testSet.size();
    string label = "";
    if (verbose)
        cout << "index\treal\tpredict" << endl;
    for (int i = 0; i < n; i++)
    {
        label = classify(testSet[i]);
        if (label == testLabel[i]) cnt++;
        if (verbose)
        {
            cout << i << "\t" << testLabel[i]
                << "\t" << label << endl;
        }
    }
    return (double)cnt / (double)n;
}

int main()
{
    KNN knn("Iris.txt", 7, 0.67, 4);
    double accuracy = knn.computeAccuracy();
    cout << "The accuary is: " << accuracy << endl;
}