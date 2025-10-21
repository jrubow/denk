/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Layer class for managing neural network layers
 * Supports forward and backpropagation and stores neuron values
 */

#include "layer.hh"

Layer::Layer(int curLayerSize, Activator &activatorRef, int prevLayerSize) :
    neurons(curLayerSize, 1), 
    weights(prevLayerSize, curLayerSize),
    activator(activatorRef) {
    this->neurons.uRandomize(1.0);
    this->weights.uRandomize(1.0);
}

Layer* Layer::createNextLayer(int newHeight, Activator &newActivator) {
    this->nextLayer = &Layer(newHeight, newActivator, this->height);
    return this->nextLayer;
}

void Layer::forward() {
    nextLayer->neurons = std::move(activator.activate(this->neurons.multiply(weights.transpose())));
}