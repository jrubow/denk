/**
 * @file leakyrelu.cc
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Leaky ReLu implementation : extension of Activator class.
 * Computes sigmoid function given input.
 */


#include "leakyrelu.hh"
#include "matrix.hh"

LeakyReLu::LeakyReLu(double alpha) {
    this->alpha = alpha;
}

double LeakyReLu::activate(double input) const {
    if (input > 0) {
        return input;
    } else {
        return this->alpha * input;
    }
}

Matrix LeakyReLu::activate(const Matrix &input) const {
    arma::mat r = input.data;
    double a = this->alpha;
    r.for_each([a](arma::mat::elem_type &x) { if (x < 0.0) x *= a; });
    Matrix result(input.getRows(), input.getCols());
    result.data = std::move(r);
    return result;
}

Matrix LeakyReLu::derivate(const Matrix &input) const {
    // derivative is 1 where x>0 else alpha. Build mask and combine.
    arma::mat pos_mask = arma::conv_to<arma::mat>::from(input.data > 0.0);
    arma::mat deriv = pos_mask + (1.0 - pos_mask) * this->alpha;
    Matrix result(input.getRows(), input.getCols());
    result.data = std::move(deriv);
    return result;
}

