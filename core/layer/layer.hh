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


#include <armadillo>
#include <vector>
#include "matrix.hh"
#include "activation.hh"

class Layer {
public:
    Matrix neurons;
    Matrix weights;
    Activation activator;

    // Constructors
    Layer(int height, Activation &activator, int weightsHeight); // randomly initializes neurons

    // Setters & Getters
    void setNeurons(Matrix neurons);
    
    // Forward Propagate
    Matrix forward(Matrix weights) const;

    // Backpropagate
    void backpropogate(Matrix weights) const;

};

#endif // LAYER_H