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
#include "layer.hh"

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void updateParameters(std::vector<Layer> &layers, std::vector<Matrix> &gradients, double learningRate) = 0;
};
#endif // OPTIMIZER_H