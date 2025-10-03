/**
 * @file optimizer.hh
 * @copyright Copyright (c) Josh Rubow (jrubow). 
 * All rights reserved.
 *
 * @brief
 * Optimizer class for optimizing model parameters.
 *
 */


#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.hh"

class Optimizer {
public:
    Optimizer();

    void updateParameters(Matrix &weights, const Matrix &gradients, double learningRate);
};
#endif // OPTIMIZER_H