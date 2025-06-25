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
#include "..\..\core\matrix\matrix.hh"
#include "..\..\core\activation\activation.hh"
// #include <memory>

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
    Matrix backward(Matrix weights) const;

};

#endif // LAYER_H