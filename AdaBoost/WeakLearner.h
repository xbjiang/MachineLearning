#include <vector>

#ifndef WEAKLEARNER_H
#define WEAKLEARNER_H

typedef std::vector< std::vector<float> > fmatrix;

class WeakLearner
{
public:
	WeakLearner() {};
	virtual ~WeakLearner() = 0;
	virtual int train(const fmatrix& X, const std::vector<float>& D) = 0;
	virtual int classify(const std::vector<float> x) = 0; 
};

#endif