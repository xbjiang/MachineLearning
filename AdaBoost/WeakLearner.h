#include <vector>

#ifndef WEAKLEARNER_H
#define WEAKLEARNER_H

typedef std::vector< std::vector<float> > fmatrix;

class WeakLearner
{
public:
	WeakLearner() {};
	// virtual ~WeakLearner() = 0; 
    // You don't need to declare destructor to be pure virtual here, because you have other pure virtual functions.
    // If you did, you need to provide a definition for it because it will always be called after the destruction of the derived part.
    ~WeakLearner() {};
    virtual int train(const fmatrix& X, const std::vector<float>& Y, const std::vector<float>& D) = 0;
	virtual float classify(const std::vector<float>& x) = 0; 
    virtual WeakLearner* clone() = 0;
    virtual const std::vector<float>& get_y_pred() = 0;
    virtual float get_error() = 0;
    virtual float test(const fmatrix& X, const std::vector<float>& Y) = 0;
};

#endif