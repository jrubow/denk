/**
 * @file momentum.cc
 * @copyright Copyright (c) Josh Rubow (jrubow). 
 * All rights reserved.
 *
 * @brief
 * Momentum class for optimizing model parameters
 * Extends Optimizer base class.
 */

#include "momentum.hh"
#include "logger.h"

Momentum::Momentum(double learningRate, double beta, std::vector<std::vector<double>> shape) {
    this->learningRate = learningRate;
    this->beta = beta;
    
    velocities = std::vector<Matrix>(shape.size());
    for (int i = 0; i < shape.size(); i++) {
        velocities[i] = Matrix(shape[i][0], shape[i][1]);
    }

    // print out the velocities for debugging
    for (size_t i = 0; i < velocities.size(); ++i) {
        std::cout << "velocities[" << i << "] (" << velocities[i].getRows() << "x" << velocities[i].getCols() << "):\n";
        for (int r = 0; r < velocities[i].getRows(); ++r) {
            for (int c = 0; c < velocities[i].getCols(); ++c) {
                std::cout << velocities[i].get(r, c);
                if (c + 1 < velocities[i].getCols()) std::cout << ' ';
            }
            std::cout << '\n';
        }
    }
}

void Momentum::updateParameters(std::vector<Layer> &layers, std::vector<Matrix> &gradients) {
    for (int i = 0; i < layers.size(); i++) {
        velocities[i] = velocities[i].scalarMultiply(beta).add(gradients[i].scalarMultiply(1 - beta));
        layers[i].weights = layers[i].weights.subtract(velocities[i].scalarMultiply(learningRate));
    }
}