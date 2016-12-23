#include "WeakLearner.h"
#include <vector>

#ifndef PLAPOCKET_H
#define PLAPOCKET_H

class PLAPocket : public WeakLearner
{
public:
    PLAPocket(float alpha = 1.0, int max_update = 100, float eps = 5e-2);
    virtual ~PLAPocket();
    virtual int train(const fmatrix& X, const std::vector<float>& Y, const std::vector<float>& D);
    virtual float classify(const std::vector<float>& x);
    virtual const std::vector<float>& get_y_pred();
private:
    float _alpha;
    int _max_update;
    float _eps;
    std::vector<float> _weight;
    std::vector<float> _y_pred; // predictions of y for training set
};
#endif