/*
* k means clustering
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>

using namespace std;

template <typename T>
class KMeans {
private:
    vector< vector<T> > dataSet;
    int row, col;
    vector< vector<T> > centroids;
    int k;
    typedef struct MinMax {
        T Min;
        T Max;
        MinMax(T min, T max) : Min(min), Max(max) {}
    } tMinMax;
    typedef struct Node {
        int centIndex;
        T minDist;
        Node(int idx, T dist) : centIndex(centIndex), minDist(dist) {}
    } infoNode;
    vector<infoNode> clusterInfo;

public:
    KMeans(char* filename, int k);
    void initCentroids();
	void initRandCentroids();
    void loadDataSet(char* filename);
    vector<int> genRandSeq(int bound);
    int randomInt(int bound);
    tMinMax getMinMax(int idx);
    void kmeans();
    T distEuclid(vector<T>& vec1, vector<T>& vec2);
	void print();
};

template <typename T>
KMeans<T>::KMeans(char* filename, int k)
{
	this->k = k;
    loadDataSet(filename);
    row = dataSet.size();
    col = dataSet[0].size();
	initCentroids();
	infoNode node(-1, -1);
	for (int i = 0; i < row; i++)
		clusterInfo.push_back(node);
}

template <typename T>
void KMeans<T>::initCentroids()
{
	vector<int> randSeq = genRandSeq(row);
	for (int i = 0; i < k; i++)
		centroids.push_back(dataSet[randSeq[i]]);
}

template <typename T>
vector<int> KMeans<T>::genRandSeq(int bound)
{
	vector<int> nums;
	// generate a sequence of 0,1,2,...,bound-1s
	for (int i = 0; i < bound; i++) 
		nums.push_back(i);
	vector<int> ret;
	int endIdx = bound - 1;
	for (int i = 0; i < k; i++)
	{
		int idx = randomInt(endIdx + 1);
		ret.push_back(nums[idx]);
		nums[idx] = nums[endIdx--];
	}
	return ret;
}

template <typename T>
int KMeans<T>::randomInt(int bound)
{
	return rand() % bound;
}

template <typename T>
void KMeans<T>::kmeans()
{
    initCentroids();
    bool clusterChanged = true;
	int iteration = 0;
    
    while (clusterChanged)
    {
		cout << "Iteration: " << iteration++ << endl;
        clusterChanged = false;
        // assign each point to its nearest centroid
        for (int i = 0; i < row; i++)
        {
            int minIndex = 0;
            T minDist = distEuclid(dataSet[i], centroids[0]);
            T dist = 0; 
            for (int j = 1; j < k; j++)
            {
                dist = distEuclid(dataSet[i], centroids[j]);
                if (dist < minDist)
                {
                    minDist = dist;
                    minIndex = j;
                }
            }

            if (clusterInfo[i].centIndex != minIndex)
            {
                clusterChanged = true;
                clusterInfo[i].centIndex = minIndex;
                clusterInfo[i].minDist = minDist; // I Just find this field useless.
            }
        }

        // update the centroids
        vector< vector<T> > newCentroids(k, vector<T>(col, 0));
        vector<int> cnt(k, 0);
        for (int i = 0; i < row; i++)
        {
            int centIdx = clusterInfo[i].centIndex;
            cnt[centIdx]++;
            for (int j = 0; j < col; j++)
            {
                newCentroids[centIdx].at(j) += dataSet[i].at(j);
            }
        }
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < col; j++)
            {
                newCentroids[i][j] /= cnt[i];
            }
        }
        centroids = move(newCentroids);
    }
}

template <typename T>
T KMeans<T>::distEuclid(vector<T>& vec1, vector<T>& vec2)
{
    if (vec1.size() != vec2.size())
    {
        cerr << "The size of the input has to be the same!" << endl;
        exit(1);
    }
        
    T sum = 0;
    for (int i = 0; i < vec1.size(); i++)
    {
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return sum;
}

template <typename T>
void KMeans<T>::loadDataSet(char* filename)
{
    ifstream in(filename);
    string buffer = "";
	// while (!in.eof()) {getline(in, buffer); ...} is wrong way of reading data from files 
	// The EOF bit will never be set. 
    // reference: http://josephmansfield.uk/articles/dont-condition-input-on-eof.html
    while (getline(in, buffer))
    {
        istringstream iss(buffer);
        T temp;
        vector<T> dataRow;
        while (iss >> temp)
            dataRow.push_back(temp);
        dataSet.push_back(dataRow);
    }
    in.close();
}

template <typename T>
void KMeans<T>::print()
{
	ofstream fout("result.txt");
	if (!fout)
	{
		cerr << "fail to open file result.txt" << endl;
		exit(1);
	}
	cout << "centroids: " << endl;
	for (int i = 0; i < k; i++)
	{
		cout << i << ": ";
		for (int j = 0; j < col; j++)
			cout << centroids[i][j] << " ";
		cout << endl;
	}
	cout << "clustering result for dataSet: " << endl;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			cout << dataSet[i][j] << " ";
			fout << dataSet[i][j] << " ";
		}
		cout << "\t" << clusterInfo[i].centIndex << endl;
		fout << "\t" << clusterInfo[i].centIndex << endl;
	}
    fout.close();
}

int main(int argc, char* argv[])
{
    /*if (argc < 3)
    {
        cout << "Usage: kmeans filename k" << endl;
        exit(1);
    }
    KMeans<double> km(argv[1], atoi(argv[2]));*/
	KMeans<double> km("./dataSet.txt", 4);
	km.kmeans();
	km.print();  
}