/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Layer class for managing neural network layers
 * Supports forward and backpropagation and stores neuron values
 */

#include "layer.hh"

Layer::Layer(int curLayerSize, Activator &activatorRef, int nextLayerSize) :
    neurons(curLayerSize, 1), 
    weights(nextLayerSize, curLayerSize),
    activator(activatorRef) {
    this->neurons.uRandomize(1.0);
    this->weights.uRandomize(1.0);
}

Matrix Layer::forward(Matrix input) {
    this->neurons = input;
    return activator.activate(weights.transpose().multiply(input));
}

void Layer::backpropogate(Matrix weights)  {
    
}

void Layer::setNeurons(Matrix input) {
    this->neurons = input;
}