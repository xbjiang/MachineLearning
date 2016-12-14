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

float SVM::kernel(int i1, int i2)
{
    float k = 2 * dot_product(_x_array[i1], _x_array[i2]);
    k -= dot_product(_x_array[i1], _x_array[i1]);
    k -= dot_product(_x_array[i2], _x_array[i2]);
    k /= 2 * _sig * _sig;
    k = exp(k);
    return k;
}

float SVM::learned_func(int k)
{
    float f = 0.0;
    for (int i = 0; i < _n; i++)
    {
        if (_alpha[i] != 0)
        {
            f += _alpha[i] * _y_array[i] 
                * dot_product(_x_array[i], _x_array[k]);
        }
    }
    f -= _b;
    return f;
}

int SVM::take_step(int i1, int i2)
{
    if (i1 == i2) return 0;
    float a1 = 0.0;
    float a2 = 0.0;
    float alpha1 = _alpha[i1];
    float alpha2 = _alpha[i2];
    float y1 = _y_array[i1];
    float y2 = _y_array[i2];
    float e1 = 0.0;
    float e2 = 0.0;

    if (alpha1 > 0 && alpha1 < _c)
        e1 = _error_cache[i1];
    else
        e1 = learned_func(i1) - y1;

    if (alpha2 > 0 && alpha2 < _c)
        e2 = _error_cache[i2];
    else
        e2 = learned_func(i2) - y2;

    float L = 0.0;
    float H = 0.0;
    float s = y1 * y2;
    float gamma = y1 + s * y2;
    if (s == 1)
    {
        L = std::max(_c, gamma);
        H = std::min(0.0f, gamma - _c);
    }
    else 
    {
        L = std::max(0.0f, -gamma);
        H = std::min(_c, _c - gamma);
    }

    if (fabs(L - H) < 1e-6) return 0;

    float K12 = kernel(i1, i2);
    float K11 = kernel(i1, i1);
    float K22 = kernel(i2, i2);
    float eta = 2 * K12 - K11 - K22;
    if (eta < 0)
    {
        a2 = alpha2 + y2 * (e2 - e1) / eta;
        if (a2 > H) a2 = H;
        else if (a2 < L) a2 = L;
    }
    else
    {
        float c1 = eta / 2.0;
        float c2 = y2 * (e1 - e2) - eta * alpha2;
        float L_obj = c1 * L * L + c2 * L;
        float H_obj = c1 * H * H + c2 * H;
        if (L_obj < H_obj - _eps)
            a2 = H_obj;
        else if (L_obj > H_obj + _eps)
            a2 = L_obj;
        else
            a2 = alpha2;
    }

    if ((a2 - alpha2) < _eps * (a2 + alpha2 + _eps))
        return 0;

    a1 = alpha1 - s * (a2 - alpha2);
    if (a1 < 0)
    {
        a1 = 0;
        a2 += s * a1;
    }
    else if (a1 > _c)
    {
        a1 = _c;
        a2 = s * (a1 - _c);
    }

    float delta_b = 0.0;
    if (alpha1 > 0 && alpha1 < _c)
    {
        delta_b = e1 + (a1 - alpha1) * y1 * K11 + (a2 - alpha2) * y2 * K12;
    }
    else if (alpha2 > 0 && alpha2 < _c)
    {
        delta_b = e2 + (a1 - alpha2) * y2 * K12 + (a2 - alpha2) * y2 * K22;
    }
    else
    {
        float b1 = e1 + (a1 - alpha1) * y1 * K11 + (a2 - alpha2) * y2 * K12;
        float b2 = e2 + (a1 - alpha2) * y2 * K12 + (a2 - alpha2) * y2 * K22;
        delta_b = (b1 + b2) / 2.0;
    }
    _b += delta_b;

    float t1 = (a1 - alpha1) * y1;
    float t2 = (a2 - alpha2) * y2;
    for (int i = 0; i < _n; i++)
    {
        if (_alpha[i] > 0 && _alpha[i] < _c)
        {
            _error_cache[i] += t1 * kernel(i1, i) + t1 * kernel(i2, i) - delta_b;
        }
    }

    _error_cache[i1] = 0.0;
    _error_cache[i2] = 0.0;

    _alpha[i1] = a1;
    _alpha[i2] = a2;
    return 1;
}