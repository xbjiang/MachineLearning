#include "WeakLearner.h"
#include <vector>

#ifndef PLAPOCKET_H
#define PLAPOCKET_H

class PLAPocket : public WeakLearner
{
public:
    PLAPocket(float alpha = 1.0, int max_update = 100, float eps = 5e-2);
    virtual ~PLAPocket() override;
    virtual int train(const fmatrix& X, const std::vector<float>& Y, const std::vector<float>& D) override;
    virtual float classify(const std::vector<float>& x) override;
    virtual PLAPocket* clone() override;
    virtual const std::vector<float>& get_y_pred() override;
    virtual float get_error() override; 
    virtual float test(const fmatrix& X, const std::vector<float>& Y) override;
private:
    float _alpha; 
    int _max_update;
    float _eps;
    std::vector<float> _weight;
    std::vector<float> _y_pred; // predictions of y for training set
    float _min_error;
};
#endif