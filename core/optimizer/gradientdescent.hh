/**
 * @file gradientdescent.hh
 * @copyright Copyright (c) Josh Rubow (jrubow). 
 * All rights reserved.
 *
 * @brief
 * Gradient descent class for optimizing model parameters.
 * Extends Optimizer base class.
 */

#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include "optimizer.hh"

class GradientDescent : public Optimizer {
public:
    GradientDescent();

    void updateParameters(Matrix &weights, const Matrix &gradients, double learningRate);
};


#endif // GRADIENTDESCENT_H