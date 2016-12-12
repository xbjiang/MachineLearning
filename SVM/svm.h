#include "svm_util.h"

#ifndef SVM_H
#define SVM_H

#define SVM_C       1.0
#define SVM_EPSILON 0.001
#define SVM_SIGMA   4.0

class SVM
{
protected:
    float _c;
    float _eps;
    float _sig;

    vector<float> _alpha;
    vector<PairArray> _x_array;
    vector<float> _y_array;
    float _b;
    vector<float> _error_cache;
    
    int _n; 
    int _n_sv; // number of support vectors

public:
    SVM();
    ~SVM();
    void train(const string& filename);
    void test(const string& filename);
    int predict(PairArray& x);

protected:
    float kernel(int i1, int i2);
    float learned_func(int k);
    int examine_example(int i1);
    int take_step(int i1, int i2);
};
#endif