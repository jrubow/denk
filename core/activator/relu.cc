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
    Matrix result = Matrix(input.getRows(), input.getCols());
    for (int r = 0; r < input.getRows(); r++) {
        for (int c = 0; c < input.getCols(); c++) {
            double val = input.get(r, c);
            if (val > 0) {
                result.set(r, c, val);
            } else {
                result.set(r, c, 0.0);
            }
        }
    }
    return result;
}

Matrix ReLu::derivate(const Matrix &input) const {
    Matrix result = Matrix(input.getRows(), input.getCols());
    for (int r = 0; r < input.getRows(); r++) {
        for (int c = 0; c < input.getCols(); c++) {
            double val = input.get(r, c);
            if (val > 0) {
                result.set(r, c, 1.0);
            } else {
                result.set(r, c, 0.0);
            }
        }
    }
    return result;
}