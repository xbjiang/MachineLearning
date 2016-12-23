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
    if (Y.size() != row || D.size() != row)
    {
        std::cerr << "Wrong dimension!" << std::endl;
        exit(1);
    }
    _weight.resize(col, 0.0);
    _y_pred.resize(row, 0.0);

    int update = 0;
    float min_loss = 1.0;
    std::vector<float> w_new(col, 0.0);
    std::vector<float> y_new(row, 0.0);
    while (update < _max_update && min_loss > _eps)
    {
        int i = rand() % row;
        if (classify(X[i]) == Y[i]) continue;
        
        update++;
        for (int j = 0; j < col; j++)
        {
            w_new[j] += _alpha * Y[i] * X[i][j] * D[i];
        }
        
        float loss = 0.0;
        for (int k = 0; k < row; k++)
        {
            y_new[k] = classify(X[k]);
            if (y_new[k] != Y[k])
                loss += D[k];
        }
        if (loss < min_loss)
        {
            min_loss = loss;
            _weight = w_new; // you can't use swap here because w_new has to used in the next iteration
            _y_pred.swap(y_new);
        }

        /*std::cerr << std::setprecision(5)
            << "iteration: " << update << "\t"
            << "loss: " << loss << "\t"
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
    for (int i = 0; i < x.size(); i++)
        s += _weight[i] * x[i];
    return s > 0 ? 1.0 : -1.0;
}

const std::vector<float>& PLAPocket::get_y_pred()
{
    return _y_pred;
}