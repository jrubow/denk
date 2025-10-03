/**
 * @file sigmoid.hh
 * @copyright Copyright (c) Josh Rubow.
 * All rights reserved.
 *
 * @brief
 * Sigmoid activation function.
 * Derived from Activation base class.
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "activation.hh"

class Sigmoid : public Activation {
public:
    Sigmoid();

    double activate(double input) const;
    Matrix activate(const Matrix &input) const;
};

#endif // SIGMOID_H