#include <iostream>
#include <map>
#include <vector>
#include <string>
using namespace std;

string postings[6][10] = {
	{ "my", "dog", "has", "flea", "problems", "help", "please", "null" },
	{ "maybe", "not", "take", "him", "to", "dog", "park", "stupid", "null" },
	{ "my", "dalmation", "is", "so", "cute", "I", "love", "him", "null" },
	{ "stop", "posting", "stupid", "worthless", "garbage", "null" },
	{ "mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him", "null" },
	{ "quit", "buying", "worthless", "dog", "food", "stupid", "null" }
};

int class_vec[6] = {0, 1, 0, 1, 0, 1};

class NaiveBayes {
    private:
		vector< vector<string> > posting_list;
		vector<int> class_list;
		map<string, int> vocab_list;
		int* return_vec;

    public:
		NaiveBayes()
		{
			vector<string> vec;
			for (int i = 0; i < 6; i++)
			{
				vec.clear();
				for (int j = 0; postings[i][j] != "null"; j++)
				{
					vec.push_back(postings[i][j]);
				}
				posting_list.push_back(vec);
			}

			for (int i = 0; i < 6; i++)
			{
				class_list.push_back(class_vec[i]);
			}
		}

		void create_vocab_list()
		{
			int index = 1;
			for (int i = 0; i < posting_list.size(); i++)
			{
				for (int j = 0; j < posting_list[i].size(); j++)
				{
					if (vocab_list.find(posting_list[i][j]) == vocab_list.end())
						vocab_list[posting_list[i][j]] = index++;
				}
			}
			int len = vocab_list.size() + 1;
			return_vec = new int[len]();

			map<string, int>::const_iterator iter = vocab_list.begin();
			cout << "vocabulary list: " << endl;
			while (iter != vocab_list.end())
			{
				cout << iter->first << ": " << iter->second << " ";
				iter++;
			}
			cout << endl;
		}

		void words2vec(int idx)
		{
			int len = vocab_list.size() + 1;
			fill(return_vec, return_vec+len, 0);
			for (int i = 0; i < posting_list[idx].size(); i++)
			{
				int index = vocab_list[posting_list[idx][i]];
				if (index != 0)
					return_vec[index] = 1;
			}
		}

		void print()
		{
			cout << "word vec: ";
			for (int i = 0; i < vocab_list.size() + 1; i++)
				cout << return_vec[i] << " ";
			cout << endl;
		}

		~NaiveBayes()
		{
			delete[] return_vec;
		}
};

int main()
{
	NaiveBayes nb;
	nb.create_vocab_list();
	nb.words2vec(2);
	nb.print();
	nb.words2vec(3);
	nb.print();
}