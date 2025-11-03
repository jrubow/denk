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
#include "layer.hh"

class SGD : public Optimizer {
public:
    SGD(double learningRate);

    void updateParameters(std::vector<Layer> &layers, std::vector<Matrix> &gradients);
};


#endif // GRADIENTDESCENT_H