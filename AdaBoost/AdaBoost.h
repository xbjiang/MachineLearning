#include "WeakLearner.h"
#include "PLAPocket.h"

#ifndef ADABOOST_H
#define ADABOOST_H

class AdaBoost
{
public:
    AdaBoost(std::size_t m, WeakLearner& wl);
    ~AdaBoost();
    int train(const fmatrix& X, const std::vector<float>& Y);
    int test(const fmatrix& X, const std::vector<float>& Y);
    float classify(const std::vector<float>& x);
private:
    std::vector<float> _alpha;          // weight for each weak learner/classifier
    std::vector<WeakLearner*> _pwls;    // array of pointers to WeakLearner
    std::size_t _M;                     // number of weak learners

    std::vector<float> _D;              // weight distribution for training set
};
#endif