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

Matrix MSE::compute(const Matrix &predicted, const Matrix &actual) const {
    return actual.subtract(predicted).toPower(2).scalarMultiply(1.0 / predicted.getRows());
}

Matrix MSE::derivate(const Matrix &actual, const Matrix &expected) const {
    return actual.subtract(expected);
}