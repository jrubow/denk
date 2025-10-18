/**
 * @file identity.hh
 * @copyright Copyright (c) Josh Rubow.
 * All rights reserved.
 *
 * @brief
 * Identity activation function.
 * Derived from Activator base class.
 */

#ifndef IDENTITY_H
#define IDENTITY_H

#include "activator.hh"

class Identity : public Activator {
public:
    Identity();

    double activate(double input) const;
    Matrix activate(const Matrix &input) const;
    Matrix derivate(const Matrix &input) const;
};

#endif // IDENTITY_H