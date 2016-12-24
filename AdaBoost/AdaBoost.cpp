#include "AdaBoost.h"
#include "math.h"
#include <numeric>
#include <stdlib.h>
#include <iostream>
#include <iomanip>

AdaBoost::AdaBoost(std::size_t m, WeakLearner& wl) : _M(m)
{
    for (std::size_t i = 0; i < m; i++)
    {
        WeakLearner* pwl = wl.clone();
        _pwls.push_back(pwl);
    }
    _alpha.resize(m, 0.0);
}

AdaBoost::~AdaBoost()
{
    for (std::size_t i = 0; i < _pwls.size(); i++)
    {
        delete _pwls[i];
    }
}

int AdaBoost::train(const fmatrix& X, const std::vector<float>& Y)
{
    size_t N = X.size();
    if (Y.size() != N)
    {
        std::cerr << "Wrong dimension!" << std::endl;
        exit(1);
    }
    _D.resize(N, 1.0 / (float)N);

    float error = 0.0;
    for (size_t m = 0; m < _M; m++)
    {
        _pwls[m]->train(X, Y, _D);
        
        error = _pwls[m]->get_error();

        _alpha[m] = 0.5 * logf( (1.0 - error) / error );

        const std::vector<float>& Y_pred = _pwls[m]->get_y_pred();

        for (size_t i = 0; i < N; i++)
        {
            _D[i] *= expf( - ( Y[i] * _alpha[m] * Y_pred[i] ) );
        }
        float sum = std::accumulate(_D.begin(), _D.end(), 0.0);
        for (size_t i = 0; i < N; i++)
        {
            _D[i] /= sum;
        }
    }
    return 0;
}

int AdaBoost::test(const fmatrix& X, const std::vector<float>& Y)
{
    std::size_t N = X.size();
    if (Y.size() != N)
    {
        std::cerr << "Wrong dimension!" << std::endl;
        exit(1);
    }
    for (size_t m = 0; m < _M; m++)
    {
        std::cerr << std::setprecision(5)
            << "weak learner " << (m + 1) << "\t"
            << "Accuracy: " << _pwls[m]->test(X, Y)
            << std::endl;
    }

    int n_correct = 0;
    for (std::size_t i = 0; i < N; i++)
    {
        float Y_pred = classify(X[i]);
        if (Y_pred == Y[i])
            n_correct++;
        /*std::cerr << i << "\t"
            << "y_real: " << Y[i] << "\t"
            << "y_pred: " << Y_pred << "\t"
            << std::endl;*/
    }
    float accuracy = (float)n_correct / (float)N;
    std::cerr << std::setprecision(5)
        << "Accuracy: " << accuracy
        << "(" << n_correct << " of " << N << ")"
        << std::endl;
    return 0;
}

float AdaBoost::classify(const std::vector<float>& x)
{
    float sum = 0.0;
    for (std::size_t m = 0; m < _M; m++)
    {
        sum += _alpha[m] * _pwls[m]->classify(x);
    }
    return sum >= 0 ? 1.0 : -1.0;
}