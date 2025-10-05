/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Matrix class for basic linear algebra operations.
 * Supports element-wise and matrix multiplication, addition, and transposition.
 * 
 * 1 Ne 10:19 For he that diligently seekteth shall find; and the mysteries of Giod shall be unfolded unto them
 * by the power of the Holy Ghost.
 * 
 */

#ifndef ACTIVATOR_H
#define ACTIVATOR_H

#include "matrix.hh"

enum ActivationType {
    SIGMOID
};

class Activator {
public:

    double activate(double input) const;
    Matrix activate(const Matrix &input) const;

    Matrix derivate(const Matrix &input) const;
};

#endif // ACTIVATOR_H