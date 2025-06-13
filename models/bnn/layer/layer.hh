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
    Matrix W; // includes bias's

    // Constructors
    Layer(int height); // Initalize weights to 0
    Layer(Matrix weights);

    // Accessors
    Matrix getWeights() const;

    // Setters
    Matrix setWeights(Matrix newWeights);
    Matrix setWeight(int x, int y, double value);
    
    // Forward Propagate
    Matrix forward(Matrix input, Activation activator) const;

    // Backpropagate
    Matrix backward(Matrix input, Activation activator) const;

};

#endif // LAYER_H