/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Identity implementation : extension of Activator class.
 * Computes identity function given input.
 */



#include "identity.hh"
#include "matrix.hh"

Identity::Identity() {}

double Identity::activate(double input) const {
    return input;
}

Matrix Identity::activate(const Matrix &input) const {
    return input;
}

Matrix Identity::derivate(const Matrix &input) const {
    return Matrix(input.getRows(), input.getCols(), 1); 
}