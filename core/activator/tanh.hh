/**
 * @file sigmoid.hh
 * @copyright Copyright (c) Josh Rubow.
 * All rights reserved.
 *
 * @brief
 * tanh activation function.
 * Derived from Activator base class.
 */

#ifndef TANH_H
#define TANH_H

#include "activator.hh"

class Tanh : public Activator {
public:
    Tanh();

    double activate(double input) const;
    Matrix activate(const Matrix &input) const;
    Matrix derivate(const Matrix &input) const;
};

#endif // TANH_H