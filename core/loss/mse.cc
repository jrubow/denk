/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Mean Squared Error class : extension of Loss class
 * Supports loss computation and derviation
 */

#include "mse.hh"
#include "matrix.hh"
#include <cmath>

MSE::MSE() {}

Matrix MSE::compute(const Matrix *expected, const Matrix *actual) const {
    return (*actual).subtract((*expected)).toPower(2).scalarMultiply(1.0 / (*expected).getRows());
}

Matrix MSE::derivate(const Matrix *actual, const Matrix *expected) const {
    Matrix diff = (*actual).subtract(*expected);
    double scale = 2.0 / (actual->getRows() * actual->getCols());
    return diff.scalarMultiply(scale);
}
