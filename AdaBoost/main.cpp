#include "../Utils/Utils.h"
#include "PLAPocket.h"
#include "WeakLearner.h"
#include <fstream>
#include <stdlib.h>
#include <iomanip>

int main(int argc, char* argv[])
{
    char* filename = "../Data/heart_scale.train";
    ifstream in(filename);
    if (!in.is_open())
    {
        std::cerr << "Cannot open file" << filename << std::endl;
        exit(1);
    }
    std::vector< std::vector<float> > X;
    std::vector<float> Y;
    read_from_kvfile(in, X, Y);
    /*for (auto x : X)
    {
        for (auto item : x)
            std::cout << item << " ";
        std::cout << std::endl;
    }*/
    int row = X.size();
    std::vector<float> D(row, 1.0 / row);
    WeakLearner* wl = new PLAPocket();
    wl->train(X, Y, D);
    const std::vector<float>& Y_pred = wl->get_y_pred();
    int n_correct = 0;
    for (std::size_t i = 0; i < Y.size(); i++)
    {
        std::cout << "y: " << Y[i] << "\t"
            << "y_pred: " << Y_pred[i] << std::endl;
        if (Y[i] == Y_pred[i])
            n_correct++;
    }
    std::cout << std::setprecision(5) << "Accuracy: "
        << ((float)n_correct / (float)row) << "%" << std::endl;
}
