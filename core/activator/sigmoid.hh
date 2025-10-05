/**
 * @file sigmoid.hh
 * @copyright Copyright (c) Josh Rubow.
 * All rights reserved.
 *
 * @brief
 * Sigmoid activation function.
 * Derived from Activator base class.
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "activator.hh"

class Sigmoid : public Activator {
public:
    Sigmoid();

    double activate(double input) const;
    Matrix activate(const Matrix &input) const;
    Matrix derivate(const Matrix &input) const;
};

#endif // SIGMOID_H