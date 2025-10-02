/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Mean Squared Error class : extension of Loss class
 * Supports loss computation and derviation
 */

#include "loss.hh"
#include <cmath>

class MSE : public Loss {
public: 
    MSE() {}

    Matrix compute(const Matrix &predicted, const Matrix &actual) const {
        return actual.subtract(predicted).toPower(2).scalarMultiply(0.5);
    }

    Matrix derivate(const Matrix &predicted, const Matrix &actual) const {
        return predicted.subtract(actual);
    }

};