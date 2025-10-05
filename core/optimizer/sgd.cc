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
#include "matrix.hh"

SGD::SGD() {

}

void updateParameters(Matrix &weights, Matrix &gradients, double learningRate) {
    weights = weights.subtract(gradients.scalarMultiply(learningRate));
}