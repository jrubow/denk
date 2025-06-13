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
    double activate(double input) {
        return 1.0 / (1.0 + std::exp(-input));
    }
};