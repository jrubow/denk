/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Layer class for managing neural network layers
 * Supports forward and backpropagation and stores neuron values
 */

#include "layer.hh"

enum InitializationType {
    UNIFORM,
    NORMAL,
    XAVIER_UNIFORM,
    XAVIER_NORMAL
};

Layer::Layer(int prevLayerSize, Activator &activatorRef, int curLayerSize, int initType) :
    neurons(curLayerSize, 1), 
    weights(curLayerSize, prevLayerSize + 1),
    activator(activatorRef) {
    this->neurons.uRandomize(1.0);
    if (initType == UNIFORM) {
        this->weights.uRandomize(1.0);
    } else if (initType == NORMAL) {

    } else if (initType == XAVIER_UNIFORM) {
        this->uxavier();
    } else if (initType == XAVIER_NORMAL) {
        this->nxavier();
    }
    this->height = curLayerSize;
}

void Layer::uxavier() {
    this->weights.uRandomize(std::sqrt(6.0 / (this->weights.getCols() + this->weights.getRows())));
}

void Layer::nxavier() {

}

Layer* Layer::createNextLayer(int newHeight, Activator &newActivator) {
    this->nextLayer = new Layer(newHeight, newActivator, this->height, UNIFORM);
    return this->nextLayer;
}

Matrix* Layer::forward(Matrix *input) {
    this->preActivation = weights.multiply(*input);
    this->neurons = std::move(activator.activate(this->preActivation));
    return &this->neurons;
}