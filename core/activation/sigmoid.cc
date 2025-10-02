/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Sigmoid implementation : extension of Activation class.
 * Computes sigmoid function given input.
 */

#include "activation.hh"
#include <cmath>

class Sigmoid : public Activation {
public:
    Sigmoid() {}

    double activate(double input) const {
        return 1.0 / (1.0 + std::exp(-input));
    }

    Matrix activate(const Matrix &input) const {
        return input.scalarMultiply(-1.0).exponent().scalarAdd(1.0).toPower(-1.0);
    }
};