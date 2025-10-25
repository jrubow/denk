/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Layer Class header file.
 * Supports forward & backward propagation and update functions.
 */

#ifndef LAYER_H
#define LAYER_H


#include <vector>
#include "matrix.hh"
#include "activator.hh"

class Layer {
public:
    Matrix neurons;
    Matrix weights;
    Activator &activator;
    Layer *nextLayer;
    int height;

    Layer(int height, Activator &activator, int weightsHeight); // randomly initializes weights
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    Layer(Layer&&) noexcept = default;
    Layer& operator=(Layer&&) noexcept = default;

    Layer* createNextLayer(int newHeight, Activator &newActivator);

    // Setters & Getters
    void setNeurons(Matrix neurons);
    
    // Forward Propagate
    Matrix* forward(Matrix *input);

};

#endif // LAYER_H