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
    int height;

    // Constructors
    Layer(int height, Activator &activator, int weightsHeight); // randomly initializes neurons


    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    Layer(Layer&&) noexcept = default;
    Layer& operator=(Layer&&) noexcept = default;

    // Setters & Getters
    void setNeurons(Matrix neurons);
    
    // Forward Propagate
    Matrix forward(Matrix weights);

    // Backpropagate
    void backpropogate(Matrix weights);

};

#endif // LAYER_H