#include "../Utils/Utils.h"
#include "PLAPocket.h"
#include "WeakLearner.h"
#include "AdaBoost.h"
#include <fstream>
#include <stdlib.h>
#include <iomanip>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: AdaBoost [num_of_learners]" << std::endl;
        exit(1);
    }
    const char* filename = "../Data/heart_scale.train";
    std::vector< std::vector<float> > X;
    std::vector<float> Y;
    load_data_set(filename, X, Y);
    
    //PLAPocket pocket(); // this is function declaration, not default initializaiton
    PLAPocket pocket(1.0, 100);
    AdaBoost adaboost(atoi(argv[1]), pocket);
    adaboost.train(X, Y);

    filename = "../Data/heart_scale.test";
    load_data_set(filename, X, Y);
    adaboost.test(X, Y);
}
