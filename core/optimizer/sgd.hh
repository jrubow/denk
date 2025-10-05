/**
 * @file gradientdescent.hh
 * @copyright Copyright (c) Josh Rubow (jrubow). 
 * All rights reserved.
 *
 * @brief
 * Stochastic Gradient Descent (SGD) class for optimizing model parameters.
 * Extends Optimizer base class.
 */

#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include "optimizer.hh"

class SGD : public Optimizer {
public:
    SGD();

    void updateParameters(Matrix &weights, const Matrix &gradients, double learningRate);
};


#endif // GRADIENTDESCENT_H