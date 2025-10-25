/**
 * @file mse.hh
 * @copyright Copyright (c) Josh Rubow (jrubow). 
 * All rights reserved.
 *
 * @brief
 * Mean Squared Error (MSE) loss function.
 * Derived from Loss base class.
 */

#ifndef MSE_H
#define MSE_H

#include "loss.hh"

class MSE : public Loss {
public:
    MSE();

    Matrix compute(const Matrix *predicted, const Matrix *actual) const;
    Matrix derivate(const Matrix *predicted, const Matrix *actual) const;
};

#endif // MSE_H