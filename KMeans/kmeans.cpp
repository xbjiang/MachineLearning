/*
* k means clustering
*/

#include <iostream>
#include <vector>

using namespace std;

template <typename T>
class KMeans {
private:
    vector< vector<T> > dataSet;
    int row, col;
    vector<T> centroids;
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
    KMeans(int k);
    void initCentroids();
	void initRandCentroids();
    void loadDataSet();
    vector<int> genRandSeq(int bound);
    int randomInt(int bound);
    tMinMax getMinMax(int idx);
    void kmeans();
	T distEuclid(vector<T> vec1, vector<T> vec2)
};

template <typename T>
KMeans<T>::KMeans(int k)
{
	this->k = k;
}

template <typename T>
void KMeans<T>::initCentroids()
{
	vector<int> randSeq = genRandSeq(row);
	for (int i = 0; i < K; i++)
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
	int endIdx = row - 1;
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

}