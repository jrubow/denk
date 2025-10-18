/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Loss class for calculating loss and its derivative
 * Supports loss computation and derviation
 */

#ifndef LOSS_H
#define LOSS_H

#include "matrix.hh"

class Loss {
public:
    virtual ~Loss() = default;

    virtual Matrix compute(const Matrix &predicted, const Matrix &actual) const = 0;
    virtual Matrix derivate(const Matrix &predicted, const Matrix &actual) const = 0;
};

#endif // LOSS_H