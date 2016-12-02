/*
* Naive Bayes Classifier
* p(ci|W) = p(W|ci)p(ci) = p(W0|ci)p(W1|ci)...p(Wn|ci)p(ci)
*/

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <numeric>
using namespace std;

string postings[6][10] = {
	{ "my", "dog", "has", "flea", "problems", "help", "please", "null" },
	{ "maybe", "not", "take", "him", "to", "dog", "park", "stupid", "null" },
	{ "my", "dalmation", "is", "so", "cute", "I", "love", "him", "null" },
	{ "stop", "posting", "stupid", "worthless", "garbage", "null" },
	{ "mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him", "null" },
	{ "quit", "buying", "worthless", "dog", "food", "stupid", "null" }
};

int class_vec[6] = {0, 1, 0, 1, 0, 1}; // 1 abusive 0 not 

class NaiveBayes {
    private:
		vector< vector<string> > posting_list;	// training set, each row for one record
		vector<int> class_list;					// label of abusive or not
		map<string, int> vocab_list;			// vocabulary set -> index
		int* return_vec;						// words vector, return value of words2vec(int idx)
		vector< vector<int> > train_mat;		// words matrix, each row for one record
		vector<float> p_vec_c0;					// p(W|c0) likelihood:¡¡frequency vector for each word in non-abusive records
		vector<float> p_vec_c1;					// p(W|c1) likelihood:¡¡frequency vector for each word in abusive records
		float p_c1;								// p(c1) prior: frequency for abusive records

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

		void words2vec(vector<string>& doc)
		{
			int len = vocab_list.size() + 1;
			fill(return_vec, return_vec+len, 0);
			for (int i = 0; i < doc.size(); i++)
			{
				int index = vocab_list[doc[i]];
				if (index != 0)
					return_vec[index] = 1;
			}
		}

		void create_train_mat()
		{
			train_mat.clear();
			for (int i = 0; i < class_list.size(); i++)
			{
				words2vec(posting_list[i]);
				train_mat.push_back(vector<int>(return_vec, return_vec + vocab_list.size() + 1));
			}
		}

		void print()
		{
			cout << "train matrix: " << endl;
			for (int i = 0; i < class_list.size(); i++)
			{
				for (int j = 0; j < vocab_list.size() + 1; j++)
					cout << train_mat[i][j] << " ";
				cout << endl;
			}
			cout << "words frequency vector: " << endl;
			cout << "p(Wi|c0): ";
			for (int i = 0; i < p_vec_c0.size(); i++)
				cout << p_vec_c0[i] << " ";
			cout << endl;
			cout << "p(Wi|c1): ";
			for (int i = 0; i < p_vec_c1.size(); i++)
				cout << p_vec_c1[i] << " ";
			cout << endl;
			cout << accumulate(p_vec_c0.begin(), p_vec_c0.end(), 0.0) << endl;
			cout << accumulate(p_vec_c1.begin(), p_vec_c1.end(), 0.0) << endl;
		}

		void trainNB()
		{
			int num_records = class_list.size();
			int num_abusive = accumulate(class_list.begin(), class_list.end(), 0);
			p_c1 = (float)num_abusive / (float)num_records;

			p_vec_c0.resize(vocab_list.size() + 1, 1.0); // words frequency vector
			p_vec_c1.resize(vocab_list.size() + 1, 1.0);
			float total_num_c0 = 2.0; // total numbers of words in non-abusive records
			float total_num_c1 = 2.0;
			for (int i = 0; i < class_list.size(); i++)
			{
				if (class_list[i] == 0)
				{
					for (int j = 0; j < train_mat[i].size(); j++)
					{
						p_vec_c0[j] += train_mat[i][j];
						total_num_c0 += train_mat[i][j];
					}
				}
				else
				{
					for (int j = 0; j < train_mat[i].size(); j++)
					{
						p_vec_c1[j] += train_mat[i][j];
						total_num_c1 += train_mat[i][j];
					}
				}
			}

			for (int i = 0; i < p_vec_c0.size(); i++)
			{
				p_vec_c0[i] = log(p_vec_c0[i] / total_num_c0);
				p_vec_c1[i] = log(p_vec_c1[i] / total_num_c1);
			}
		}

		int classify(vector<string> doc)
		{
			words2vec(doc);
			/*float p0 = inner_product(p_vec_c0.begin() + 1, p_vec_c0.end(), return_vec + 1, 0)
				+ log(1 - p_c1);
			float p1 = inner_product(p_vec_c1.begin() + 1, p_vec_c0.end(), return_vec + 1, 0)
				+ log(p_c1);*/
			float p0 = 0.0;
			float p1 = 0.0;
			for (int i = 0; i < p_vec_c0.size(); i++)
			{
				if (return_vec[i] == 1)
				{
					p0 += p_vec_c0[i];
					p1 += p_vec_c1[i];
				}
			}
			p0 += log(1.0 - p_c1);
			p1 += log(p_c1);
			cout << "p0: " << p0 << endl;
			cout << "p1: " << p1 << endl;
			if (p0 > p1) return 0;
			return 1;
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
	nb.create_train_mat();
	nb.trainNB();
	nb.print();

	vector<string> doc1 = {"love", "my", "dalmation"};
	vector<string> doc2 = {"stupid", "garbage"};
	cout << "doc1 classified as:" << nb.classify(doc1) << endl;
	cout << "doc2 classified as:" << nb.classify(doc2) << endl;
}