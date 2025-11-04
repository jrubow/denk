/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * ReLu implementation : extension of Activator class.
 * Computes sigmoid function given input.
 */



#include "relu.hh"
#include "matrix.hh"

ReLu::ReLu() {}

double ReLu::activate(double input) const {
    if (input > 0) {
        return input;
    } else {
        return 0.0;
    }
}

Matrix ReLu::activate(const Matrix &input) const {
    arma::mat r = input.data;
    r.for_each([](arma::mat::elem_type &x) { if (x < 0.0) x = 0.0; });
    Matrix result(input.getRows(), input.getCols());
    result.data = std::move(r);
    return result;
}

Matrix ReLu::derivate(const Matrix &input) const {
    // derivative is 1 where input > 0, else 0. Use vectorized ops.
    arma::mat mask = arma::conv_to<arma::mat>::from(input.data > 0.0);
    Matrix result(input.getRows(), input.getCols());
    result.data = std::move(mask);
    return result;
}