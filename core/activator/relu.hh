/**
 * @file relu.hh
 * @copyright Copyright (c) Josh Rubow.
 * All rights reserved.
 *
 * @brief
 * ReLu activation function.
 * Derived from Activator base class.
 */

#ifndef RELU_H
#define RELU_H

#include "activator.hh"

class ReLu : public Activator {
public:
    ReLu();

    double activate(double input) const;
    Matrix activate(const Matrix &input) const;
    Matrix derivate(const Matrix &input) const;
};

#endif // RELU_H