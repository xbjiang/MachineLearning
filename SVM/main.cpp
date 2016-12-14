#include "svm.h"

int main()
{
    SVM svm;
    //svm.train("heart_scale.train", "heart_scale.model");
    svm.test("heart_scale.test", "heart_scale.model");
}