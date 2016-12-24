#include "PLAPocket.h"
#include <iostream>
#include <stdlib.h>
#include <iomanip>

PLAPocket::PLAPocket(float alpha, int max_update, float eps)
    : _alpha(alpha), _max_update(max_update), _eps(eps)
{
}

PLAPocket::~PLAPocket()
{
}

int PLAPocket::train(const fmatrix& X, const std::vector<float>& Y, 
    const std::vector<float>& D)
{
    std::size_t row = X.size();
    std::size_t col = X[0].size();
    /*if (Y.size() != row || D.size() != row)
    {
        std::cerr << "Wrong dimension!" << std::endl;
        exit(1);
    }*/ // parameter check should be done in the outermost layer
    _weight.resize(col, 0.0);
    _y_pred.resize(row, 0.0);
    _min_error = 1.0;

    int update = 0;
    float min_loss = 1.0;
    std::vector<float> w_new(col, 0.0);
    std::vector<float> y_new(row, 0.0);
    while (update < _max_update && min_loss > _eps)
    {
        int i = rand() % row;
        if (classify(X[i]) == Y[i]) continue;
        
        update++;
        for (std::size_t j = 0; j < col; j++)
        {
            w_new[j] += _alpha * Y[i] * X[i][j] * D[i];
        }
        
        float error = 0.0;
        for (std::size_t k = 0; k < row; k++)
        {
            y_new[k] = classify(X[k]);
            if (y_new[k] != Y[k])
                error += D[k];
        }
        if (error < _min_error)
        {
            _min_error = error;
            _weight = w_new; // you can't use swap here because w_new has to used in the next iteration
            _y_pred.swap(y_new);
        }

        /*std::cerr << std::setprecision(5)
            << "iteration: " << update << "\t"
            << "error: " << error << "\t"
            << "i: " << i
            << std::endl;*/
    }
    return 0;
}

float PLAPocket::classify(const std::vector<float>&x)
{
    if (x.size() != _weight.size())
    {
        std::cerr << "Wrong dimension!" << std::endl;
        exit(1);
    }
    float s = 0.0;
    for (std::size_t i = 0; i < x.size(); i++)
        s += _weight[i] * x[i];
    return s >= 0 ? 1.0 : -1.0;
}

PLAPocket* PLAPocket::clone()
{
    return new PLAPocket(*this);
}

const std::vector<float>& PLAPocket::get_y_pred()
{
    return _y_pred;
}

float PLAPocket::get_error()
{
    return _min_error;
}

float PLAPocket::test(const fmatrix& X, const std::vector<float>& Y)
{
    std::size_t N = X.size();
    if (Y.size() != N)
    {
        std::cerr << "Wrong dimension!" << std::endl;
        exit(1);
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
    /*std::cerr << std::setprecision(5)
        << "Accuracy: " << accuracy << "%"
        << "(" << n_correct << " of " << N << ")"
        << std::endl;*/
    return accuracy;
}