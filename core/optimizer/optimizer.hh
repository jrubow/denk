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
    Optimizer();

    void updateParameters(std::vector<Layer> &layers, std::vector<Matrix> &gradients, double learningRate);
};
#endif // OPTIMIZER_H