/**
 * @file gradientdescent.hh
 * @copyright Copyright (c) Josh Rubow (jrubow). 
 * All rights reserved.
 *
 * @brief
 * Newton's Method class for optimizing model parameters
 * Extends Optimizer base class.
 */

#ifndef NEWTONSMETHOD_HH
#define NEWTONSMETHOD_HH

#include "optimizer.hh"
#include "layer.hh"

class NewtonsMethod : public Optimizer {
public:
    NewtonsMethod();

    void updateParameters(Matrix &weights, const Matrix &gradients, double learningRate);
};

#endif // NEWTONSMETHOD_HH
