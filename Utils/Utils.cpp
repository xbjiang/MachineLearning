#include "Utils.h"

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

int read_from_kvfile(ifstream& is, vector< vector<float> >& X, vector<float>& Y)
{
    string itemLine = "";
    std::size_t maxSize = 0;
    while (getline(is, itemLine))
    {
        vector<string> items;
        split(itemLine, " ", items);
        Y.push_back(atof(items[0].c_str()));

        vector<float> itemVec;
        for (std::size_t i = 1; i < items.size(); i++)
        {
            vector<string> kv;
            split(items[i], ":", kv);
            int k = atoi(kv[0].c_str());
            float v = atof(kv[1].c_str());
            if (itemVec.size() < k - 1)
                itemVec.resize(k - 1, 0.0);
            itemVec.push_back(v);
        }
        if (itemVec.size() > maxSize)
            maxSize = itemVec.size();
        X.push_back(itemVec);
    }

    for (std::size_t i = 0; i < X.size(); i++)
    {
        if (X[i].size() < maxSize)
            X[i].resize(maxSize, 0.0);
    }
    return 0;
}

void load_data_set(const string& filename, vector< vector<float> >& X, vector<float>& Y)
{
    ifstream in(filename);
    if (!in.is_open())
    {
        std::cerr << "Cannot open file" << filename << std::endl;
        exit(1);
    }
    X.clear();
    Y.clear();
    read_from_kvfile(in, X, Y);
}