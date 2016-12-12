#include <math.h>
#include "svm_util.h"
#include "svm.h"

SVM::SVM()
{
    _c = SVM_C;
    _eps = SVM_EPSILON;
    _sig = SVM_SIGMA;

    _b = 0.0;
    _n = 0;
    _n_sv = 0;
}

void SVM::train(const string& filename)
{
    _alpha.clear();
    _x_array.clear();
    _y_array.clear();
    _error_cache.clear();

    load_data_set(filename, _x_array, _y_array);
    _n = _x_array.size();
    _b = 0.0;
    _alpha.resize(_n, 0.0);
    _error_cache.resize(_n, 0.0);

    int num_changed = 0;
    int examine_all = 1;
    while (num_changed > 0 || examine_all)
    {
        num_changed = 0;
        if (examine_all)
        {
            for (int i = 0; i < _n; i++)
                num_changed += examine_example(i);
        }
        else
        {
            for (int i = 0; i < _n; i++)
            {
                if (_alpha[i] > 0 && _alpha[i] < _c) // TODO is this the right way to compare two floats, do some research!
                    num_changed += examine_example(i);
            }
        }

        if (examine_all)
            examine_all = 0; 
        else if (num_changed == 0)
            examine_all = 1;
    }
}

int SVM::examine_example(int i1)
{
    float alpha1 = _alpha[i1];
    float y1 = _y_array[i1];
    float e1 = 0;

    if (alpha1 > 0 && alpha1 < _c)
    {
        e1 = _error_cache[i1];
    }
    else
    {
        e1 = learned_func(i1) - y1;
    }

    float r1 = y1 * e1;

    if ((alpha1 > 0 && r1 > _eps) || (alpha1 < _c && r1 < -_eps))
    {
        float max_error = 0.0;
        float e2 = 0.0;
        int i2 = -1;
        for (int i = 0; i < _n; i++)
        {
            if (_alpha[i] > 0 && _alpha[i] < _c)
            {
                e2 = _error_cache[i];
                float temp = fabs(e1 - e2);
                if (temp > max_error)
                {
                    max_error = temp;
                    i2 = i;
                }
            }
        }
        if (i2 >= 0)
        {
            if (take_step(i1, i2))
                return 1;
        }

        int rand_start = rand() % _n;
        for (int i = rand_start; i < _n + rand_start; i++)
        {
            i2 = i % _n;
            if (_alpha[i2] > 0 && _alpha[i2] < _c)
            {
                if (take_step(i1, i2))
                    return 1;
            }
        }

        rand_start = rand() % _n;
        for (int i = rand_start; i < _n + rand_start; i++)
        {
            i2 = i % _n;
            if (take_step(i1, i2))
                return 1;
        }
    }
    return 0;
}

