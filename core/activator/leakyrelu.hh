/**
 * @file leakyrelu.hh
 * @copyright Copyright (c) Josh Rubow.
 * All rights reserved.
 *
 * @brief
 * Leaky ReLu activation function.
 * Derived from Activator base class.
 */

#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "activator.hh"

class LeakyReLu : public Activator {
public:
    double alpha;

    LeakyReLu(double alpha);

    double activate(double input) const;
    Matrix activate(const Matrix &input) const;
    Matrix derivate(const Matrix &input) const;
};

#endif // LEAKYRELU_H