#include <string>
#include <map>
#include <set>
#include <vector>
#include <math.h>
#include <iostream>

using namespace std;

#define N 14
#define feature 4

map<string, vector<string>> X;
string x[N][feature+1] = 
{
	{ "Sunny", "Hot", "High", "Weak", "no" },    
	{ "Sunny", "Hot", "High", "Strong", "no" },
	{ "Overcast", "Hot", "High", "Weak", "yes" },
	{ "Rain", "Mild", "High", "Weak", "yes" },
	{ "Rain", "Cool", "Normal", "Weak", "yes" },
	{ "Rain", "Cool", "Normal", "Strong", "no" },
	{ "Overcast", "Cool", "Normal", "Strong", "yes" },
	{ "Sunny", "Mild", "High", "Weak", "no" },
	{ "Sunny", "Cool", "Normal", "Weak", "yes" },
	{ "Rain", "Mild", "Normal", "Weak", "yes" },
	{ "Sunny", "Mild", "Normal", "Strong", "yes" },
	{ "Overcast", "Mild", "High", "Strong", "yes" },
	{ "Overcast", "Hot", "Normal", "Weak", "yes" },
	{ "Rain", "Mild", "High", "Strong", "no" },
};

string attributes[] = { "Outlook", "Temperature", "Humidity", "Wind", "label" }; // label for play tennis or not

// 将二维数据转换为从attribute到value向量的map
void createDataSet()
{
	X.clear();
	for (int i = 0; i < feature + 1; i++)
	{
		X.insert(pair<string, vector<string> >(attributes[i], vector<string>(N)));
		for (int j = 0; j < N; j++)
			X[attributes[i]][j] = x[j][i];
	}
}

// 计算信息熵
double calcEntropy(map<string, vector<string> >& data)
{
	map<string, int> classCount;
	
	for (int i = 1; i < data["label"].size(); i++)
	{
		classCount[data["label"][i]] += 1;
 	}
	double result = 0;
	for (auto entry : classCount)
	{
		double ratio = (double)entry.second / (data["label"].size());
		result -= ratio * log(ratio) / log(2);
	}
	return result;
}

//按照指定特征划分数据集
map<string, vector<string>> splitData(map<string, vector<string> >& data, string attribute, string fVal)
{
	map<string, vector<string> > ret;
	for (auto& entry : data)
	{
		//初始化，ret为以attribute为键，attribute value为值的map
		//不包括划分的属性
		if (entry.first != attribute)
			ret[entry.first] = vector<string>(); 
	}
	for (int i = 0; i < data[attribute].size(); i++)
	{
		//找出指定attribute value的数据
		if (data[attribute][i] == fVal)
		{
			for (auto& entry : data)
			{
				if (entry.first != attribute)
					ret[entry.first].push_back(entry.second[i]);
			}
		}
	}
	return ret;
}

//根据指定特征生成特征取值列表
vector<string> createFeatureValueList(map<string, vector<string> >& data, string attribute)
{
	set<string> aSet;
	for (int i = 1; i < data[attribute].size(); i++)
		aSet.insert(data[attribute][i]);
	vector<string> ret;
	for (auto entry : aSet)
		ret.push_back(entry);
	return ret;
}

// 找出最优划分属性
string chooseBestFeatureToSplit(map<string, vector<string> >& data)
{
	double currentEntropy = calcEntropy(data);
	double maxInfoGain = 0;
	double infoGain = 0;
	string bestFeature = "";
	for (auto& entry : data)
	{
		if (entry.first == "label")
			continue;
		double nextEntropy = 0;
		vector<string> featureValueList = createFeatureValueList(data, entry.first);
		for (string fVal : featureValueList)
		{
			map<string, vector<string> > subData = splitData(data, entry.first, fVal);
			double ratio = (double)subData["label"].size() / (double)data["label"].size();
			nextEntropy += ratio * calcEntropy(subData);
		}
		infoGain = currentEntropy - nextEntropy;
		if (infoGain > maxInfoGain) {
			maxInfoGain = infoGain;
			bestFeature = entry.first;
		}
	}
	return bestFeature;
}

// 返回出现次数最多的分类名称
string majorityCnt(vector<string>& classList)
{
	map<string, int> cntMap;
	for (int i = 0; i < classList.size(); i++)
	{
		cntMap[classList[i]]++;
	}
	string result = "";
	int maxCnt = 0;
	map<string, int>::iterator it;
	for (it = cntMap.begin(); it != cntMap.end(); ++it)
	{
		if (it->second > maxCnt)
		{
			maxCnt = it->second;
			result = it->first;
		}
	}
	return result;
}

struct Node
{
	string attribute;
	string val;
	bool isLeaf;
	vector<Node*> childs;
	Node()
	{
		attribute = "";
		val = "";
		isLeaf = false;
	}
};

Node* root = NULL;

Node* createTree(Node* root, map<string, vector<string> >& data)
{
	vector<string> classList;
	set<string> classSet;
	if (root == NULL)
		root = new Node();

	for (string& entry : data["label"])
	{
		classList.push_back(entry);
		classSet.insert(entry);
	}

	if (classSet.size() == 1)
	{
		root->isLeaf = true;
		root->attribute = *classSet.begin();
		return root;
	}

	if (data.size() == 1)
	{
		root->isLeaf = true;
		root->attribute = majorityCnt(classList);
		return root;
	}
	
	string bestFeature = chooseBestFeatureToSplit(data);
	vector<string> featureValueList = createFeatureValueList(data, bestFeature);
	root->attribute = bestFeature;
	
	for (int i = 0; i < featureValueList.size(); i++)
	{
		Node* newNode = new Node();
		createTree(newNode, splitData(data, bestFeature, featureValueList[i]));
		newNode->val = featureValueList[i];
		root->childs.push_back(newNode);
	}
	return root;
}

void print(Node* root, int depth)
{
	for (int i = 0; i < depth; i++)
		cout << "\t";

	if (root->val != "")
	{
		cout << root->val << endl;
		for (int i = 0; i < depth + 1; i++)
			cout << "\t";
	}
	cout << root->attribute << endl;
	vector<Node*>::iterator it;
	for (it = root->childs.begin(); it != root->childs.end(); ++it)
	{
		print(*it, depth+1);
	}
}

int main()
{
	createDataSet();
	root = createTree(root, X);
	print(root, 0);
}





