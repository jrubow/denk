/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Layer class for managing neural network layers
 * Supports forward and backpropagation and stores neuron values
 */

#include "layer.hh"

Layer::Layer(int prevLayerSize, Activator &activatorRef, int curLayerSize) :
    neurons(curLayerSize, 1), 
    weights(curLayerSize, prevLayerSize + 1),
    activator(activatorRef) {
    this->neurons.uRandomize(1.0);
    this->weights.uRandomize(1.0);
}

Layer* Layer::createNextLayer(int newHeight, Activator &newActivator) {
    this->nextLayer = new Layer(newHeight, newActivator, this->height);
    return this->nextLayer;
}

Matrix* Layer::forward(Matrix *input) {
    this->neurons = std::move(activator.activate(weights.multiply(*input)));
    return &this->neurons;
}