#include "svm_util.h"
using std::ifstream;
using std::cerr;
using std::endl;
using std::make_pair;

vector<string>& split(const string& str, const string& delims, vector<string>& ret)
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
	if (prev != str.size()) ret.emplace_back(str, prev, str.size() - prev);
	return ret;
}

bool cmp(const pair<int, float>& lhs, const pair<int, float>& rhs)
{
	return lhs.first < rhs.first;
}

void load_data_set(const string& filename, vector<PairArray>& x, vector<float>& y)
{
	ifstream fin(filename);
	if (!fin.is_open())
	{
		cerr << "Open file " << filename << "failed!" << endl;
		exit(0);
	}

	string itemLine = "";
	while (getline(fin, itemLine))
	{
		vector<string> items;
		split(itemLine, " ", items);
		y.push_back(atof(items[0].c_str()));

		PairArray pArr;
		for (int i = 1; i < items.size(); i++)
		{
			vector<string> pairItem;
			split(items[i], ":", pairItem);
			int idx = atoi(pairItem[0].c_str());
			float val = atof(pairItem[1].c_str());
			pArr.push_back(make_pair(idx, val));
		}
		sort(pArr.begin(), pArr.end(), cmp); // no need to add std:: ? why?
		x.push_back(pArr);
	}
}

float dot_product(const PairArray& arr1, const PairArray& arr2)
{
	int p1 = 0; 
	int p2 = 0;
	int a1 = -1;
	int a2 = -1;
	float dot = 0.0;
	while (p1 < arr1.size() && p2 < arr2.size())
	{
		a1 = arr1[p1].first;
		a2 = arr2[p2].first;
		if (a1 == a2)
		{
			dot += arr1[p1].second * arr2[p2].second;
			p1++;
			p2++;
		}
		else if (p1 < p2)
			p1++;
		else
			p2++;
	}
	return dot;
}

int main()
{
	vector<PairArray> x;
	vector<float> y;
	load_data_set("heart_scale.test", x, y);
	for (int i = 0; i < x.size(); i++)
	{
		std::cout << y[i] << " ";
		for (int j = 0; j < x[i].size(); j++)
		{
			std::cout << x[i][j].first << ":"
				<< x[i][j].second << " ";
		}
		std::cout << endl;
	}

	std::cout << dot_product(x[0], x[0]) << endl;
}