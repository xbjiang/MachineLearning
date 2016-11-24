#include <string>
#include <map>
#include <set>
#include <vector>
#include <math.h>

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

// ����ά����ת��Ϊ��attribute��value������map
void createDateSet()
{
	for (int i = 0; i < feature + 1; i++)
	{
		X[attributes[i]] = vector<string>(N);
		for (int j = 0; j < N; j++)
			X[attributes[i]][j] = x[j][i];
	}
}

// ������Ϣ��
double calcEntropy(map<string, vector<string>>& data)
{
	map<string, int> classCount;
	
	for (int i = 1; i < data["label"].size(); i++)
	{
		classCount[data["label"][i]] += 1;
 	}
	double result = 0;
	for (auto entry : classCount)
	{
		double ratio = (double)entry.second / (data.size() - 1);
		result -= ratio * log(ratio) / log(2);
	}
	return result;
}

//����ָ�������������ݼ�
map<string, vector<string>> splitData(map<string, vector<string>>& data, string attribute, string fVal)
{
	map<string, vector<string>> ret;
	for (auto& entry : data)
	{
		//��ʼ����retΪ��attributeΪ����attribute valueΪֵ��map
		//���������ֵ�����
		if (entry.first != attribute)
			ret[entry.first] = vector<string>(); 
	}
	for (int i = 0; i < data[attribute].size(); i++)
	{
		//�ҳ�ָ��attribute value������
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

//����ָ��������������ȡֵ�б�
vector<string> createFeatureValueList(map<string, vector<string>>& data, string attribute)
{
	set<string> aSet;
	for (int i = 1; i < data[attribute].size(); i++)
		aSet.insert(data[attribute][i]);
	vector<string> ret;
	for (auto entry : aSet)
		ret.push_back(entry);
	return ret;
}

// �ҳ����Ż�������
string chooseBestFeatureToSplit(map<string, vector<string>>& data)
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
			map<string, vector<string>> subData = splitData(data, entry.first, fVal);
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