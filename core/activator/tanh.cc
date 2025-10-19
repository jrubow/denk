/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Sigmoid implementation : extension of Activator class.
 * Computes sigmoid function given input.
 */

#include "tanh.hh"
#include "matrix.hh"

Tanh::Tanh() {}

double Tanh::activate(double input) const {
    return std::tanh(input);
}

Matrix Tanh::activate(const Matrix &input) const {
    return input.exponent().subtract(input.scalarMultiply(-1).exponent())
                .multiplyElementwise(input.exponent().add(input.scalarMultiply(-1).exponent()).toPower(-1));
}

Matrix Tanh::derivate(const Matrix &input) const {
    return this->activate(input).toPower(2).scalarAdd(-1).scalarMultiply(-1);
}
