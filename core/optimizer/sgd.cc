/**
 * @file gradientdescent.cc
 * @copyright Copyright (c) Josh Rubow (jrubow). 
 * All rights reserved.
 *
 * @brief
 * Gradient descent class for optimizing model parameters.
 * Extends Optimizer base class.
 */

#include "sgd.hh"
#include "layer.hh"
#include "matrix.hh"

SGD::SGD() {

}

void updateParameters(std::vector<Layer> &layers, std::vector<Matrix> &gradients, double learningRate) {
    // add error if weights != gradients
    
    for (int i = 0; i < layers.size(); i++) {
        layers[i].weights = layers[i].weights.subtract(gradients[i].scalarMultiply(learningRate));
    }
}