/*
* SVM 
* Why are there so many support vectors? Is there a bug?
*/

#include "svm_util.h"
#include "svm.h"
using std::ifstream;
using std::ofstream;

SVM::SVM()
{
    _c = SVM_C;
    _eps = SVM_EPSILON;
    _sig = SVM_SIGMA;

    _b = 0.0;
    _n = 0;
    _n_sv = 0;
}

void SVM::test(const string& fname_test, const string& fname_model)
{
    _x_array.clear();
    _y_array.clear();
    _alpha.clear();
    ifstream is_model(fname_model);
    load_model(is_model);

    ifstream is_test(fname_test);
    load_data_set(is_test, _x_array, _y_array); // load test set

    _n = _x_array.size();

    int n_correct = 0;
    int y_pred = 0;
    for (int i = _n_sv; i < _n; i++)
    {
        float f = 0;
        for (int j = 0; j < _n_sv; j++)
        {
            f += _alpha[j] * _y_array[j] * kernel(j, i);
        }
        f -= _b;
        y_pred = f >= 0 ? 1.0 : -1.0;
        std::cerr << "y_real: " << _y_array[i] << "\t"
            << "y_pred: " << y_pred << std::endl;
        if ((y_pred > 0 && _y_array[i] > 0)
            || (y_pred < 0 && _y_array[i])) // why don't I just use (y_pred == _y_array[i])? Again, you need to do some research on float comparison! 
        {
            n_correct++;
        }
    }
    std::cerr << std::setprecision(5)
        << "Accuracy: " << 100.0 * n_correct / (_n - _n_sv)
        << "% (" << n_correct << "/" << (_n - _n_sv) << ")"
        << std::endl;
}

int SVM::load_model(ifstream& is)
{
    is >> _b;
    is >> _n_sv;
    _alpha.resize(_n_sv, 0.0);
    for (int i = 0; i < _n_sv; i++)
        is >> _alpha[i];
    is.ignore(); // ignore an "\n"
    load_data_set(is, _x_array, _y_array);
    return 0;
}

void SVM::train(const string& fname_train, const string& fname_model)
{
    _alpha.clear();
    _x_array.clear();
    _y_array.clear();
    _error_cache.clear();

    ifstream is(fname_train);
    load_data_set(is, _x_array, _y_array);
    _n = _x_array.size();
    _b = 0.0;
    _alpha.resize(_n, 0.0);
    _error_cache.resize(_n, 0.0);

    int num_changed = 0;
    int examine_all = 1;
    int iter_num = 0;
    while (num_changed > 0 || examine_all)
    {
        iter_num++;
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

        float s = 0.0;
        float t = 0.0;
        float obj = 0.0;
        for (int i = 0; i < _n; i++)
        {
            s += _alpha[i];
        }

        for (int i = 0; i < _n; i++)
        {
            for (int j = 0; j < _n; j++)
            {
                t += _alpha[i] * _alpha[j] * _y_array[i] * _y_array[j] * kernel(i, j);
            }
        }
        obj = s - t / 2.0;
        std::cerr << std::setprecision(5) 
            << "Iteration: " << iter_num << "\t"
            << "examine_all: " << examine_all << "\t"
            << "num_changed: " << num_changed << "\t"
            << "Objective func: " << obj << "\t"
            << "Error rate: " << error_rate() 
            << std::endl;

        if (examine_all)
            examine_all = 0;
        else if (num_changed == 0)
            examine_all = 1;
    }

    ofstream os(fname_model);
    dump_model(os);
}

int SVM::dump_model(ofstream& os)
{
    os << _b << std::endl;
    _n_sv = 0;
    vector<int> idx_vec;
    for (int i = 0; i < _n; i++)
    {
        if (_alpha[i] > 0)
        {
            _n_sv++;
            idx_vec.push_back(i);
        }
    }
    os << _n_sv << std::endl;
    for (int i = 0; i < idx_vec.size(); i++)
    {
        int idx = idx_vec[i];
        os << _alpha[idx] << std::endl;
    }
    for (int i = 0; i < idx_vec.size(); i++)
    {
        string s;
        int idx = idx_vec[i];
        write_sample(s, _x_array[idx], _y_array[idx]);
        os << s << std::endl;
    }
    return 0;
}

float SVM::error_rate()
{
    int n_error = 0;
    for (int i = 0; i < _n; i++)
    {
        if ((learned_func(i) >= 0 && _y_array[i] < 0)
            || (learned_func(i) < 0 && _y_array[i] > 0))
        {
            n_error++;
        }
    }
    return 1.0 * n_error / _n;
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
        if (_alpha[i] > 0)
        {
            f += _alpha[i] * _y_array[i] * kernel(i, k);
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
    float gamma = alpha1 + s * alpha2;
    if (s == 1)
    {
        L = std::max(0.0f, gamma - _c);
        H = std::min(_c, gamma);
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
            a2 = H;
        else if (L_obj > H_obj + _eps)
            a2 = L;
        else
            a2 = alpha2;
    }

    if (fabs(a2 - alpha2) < _eps * (a2 + alpha2 + _eps))
        return 0;

    a1 = alpha1 - s * (a2 - alpha2);
    if (a1 < 0)
    {
        a2 += s * a1; // compute a2 before a1, or store (0-a1) first
        a1 = 0;
    }
    else if (a1 > _c) 
    {
        a2 += s * (a1 - _c);
        a1 = _c;
    }

    float delta_b = 0.0;
    if (a1 > 0 && a1 < _c)
    {
        delta_b = e1 + (a1 - alpha1) * y1 * K11 + (a2 - alpha2) * y2 * K12;
    }
    else if (a2 > 0 && a2 < _c)
    {
        delta_b = e2 + (a1 - alpha1) * y1 * K12 + (a2 - alpha2) * y2 * K22;
    }
    else
    {
        float b1 = e1 + (a1 - alpha1) * y1 * K11 + (a2 - alpha2) * y2 * K12;
        float b2 = e2 + (a1 - alpha1) * y1 * K12 + (a2 - alpha2) * y2 * K22;
        delta_b = (b1 + b2) / 2.0;
    }
    _b += delta_b;

    float t1 = (a1 - alpha1) * y1;
    float t2 = (a2 - alpha2) * y2;
    for (int i = 0; i < _n; i++)
    {
        if (_alpha[i] > 0 && _alpha[i] < _c)
        {
            _error_cache[i] += t1 * kernel(i1, i) + t2 * kernel(i2, i) - delta_b;
        }
    }

    _error_cache[i1] = 0.0;
    _error_cache[i2] = 0.0;

    _alpha[i1] = a1;
    _alpha[i2] = a2;
    return 1;
}

SVM::~SVM() {}