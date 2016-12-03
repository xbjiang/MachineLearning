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
    vector<T> centeroids;
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
    void loadDataSet();
    vector<int> genRandSeq(int bound);
    int RandomInt(int bound);
    tMinMax getMinMax(int idx);
    void kmeans();
};