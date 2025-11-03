/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Sigmoid implementation : extension of Activator class.
 * Computes sigmoid function given input.
 */



#include "sigmoid.hh"
#include "matrix.hh"

Sigmoid::Sigmoid() {}

double Sigmoid::activate(double input) const {
    return 1.0 / (1.0 + std::exp(-input));
}

Matrix Sigmoid::activate(const Matrix &input) const {
    return input.scalarMultiply(-1.0).exponent().scalarAdd(1.0).toPower(-1.0);
}

Matrix Sigmoid::derivate(const Matrix &input) const {
    return this->activate(input).multiplyElementwise(
            this->activate(input).scalarAdd(-1.0).scalarMultiply(-1.0)); 
}