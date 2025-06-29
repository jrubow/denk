/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Layer class for managing neural network layers
 * Supports forward and backpropagation and stores neuron values
 */

#include "layer.hh"
#include <armadillo>

Layer::Layer(int height, Activation &activator, int weightsHeight) :
    neurons(height, 1), weights(0, weightsHeight) {
    this->neurons.uRandomize(1.0);
    this->weights.uRandomize(1.0);
    this->activator = activator;
}

Matrix Layer::forward(Matrix weights) const {
    return activator.activate(weights.multiply(neurons));
}

void Layer::setNeurons(Matrix input) {
    this->neurons = input;
}