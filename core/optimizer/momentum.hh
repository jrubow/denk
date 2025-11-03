/**
 * @file momentum.hh
 * @copyright Copyright (c) Josh Rubow (jrubow). 
 * All rights reserved.
 *
 * @brief
 * Momentum class for optimizing model parameters
 * Extends Optimizer base class.
 */

#ifndef MOMENTUM_HH
#define MOMENTUM_HH

#include "optimizer.hh"

class Momentum : public Optimizer {
public:
    double beta;
    std::vector<Matrix> velocities;
    double learningRate;

    Momentum(double learningRate, double beta, std::vector<std::vector<double>> shape);

    void updateParameters(std::vector<Layer> &layers, std::vector<Matrix> &gradients);
};

#endif //MOMENTUM_HH
