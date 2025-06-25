/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Matrix class for basic linear algebra operations.
 * Supports element-wise and matrix multiplication, addition, and transposition.
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../matrix/matrix.hh"

enum ActivationType {
    SIGMOID
};

class Activation {
public:
    Activation();
    double activate(double input) const;
    Matrix activate(Matrix input) const;
};

#endif // ACTIVATION_H