/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Sigmoid implementation : extension of Activation class.
 * Computes sigmoid function given input.
 */

#ifndef SIGMOID_C
#define SIGMOID_C

#include "activation.hh"
#include <cmath>

class Sigmoid : public Activation {
public:
    Sigmoid() {}

    double activate(double input) {
        return 1.0 / (1.0 + std::exp(-input));
    }
};

#endif // SIGMOID_C