/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Matrix class for basic linear algebra operations.
 * Supports element-wise and matrix multiplication, addition, and transposition.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <armadillo>
#include <vector>
#include <iostream>

class Matrix {
public:
    arma::mat data;

    // Constructors
    Matrix(int r, int c);
    Matrix(int r, int c, const std::vector<double>& initial_data);

    // Accessors for dimensions
    int getRows() const;
    int getCols() const;

    // Accessors for elements
    double get(int r, int c) const;
    void set(int r, int c, double value);

    // Basic operations
    Matrix add(const Matrix& mat) const;
    Matrix subtract(const Matrix& mat) const;
    Matrix multiplyElementwise(const Matrix& mat) const;
    Matrix multiply(const Matrix& mat) const;
    Matrix transpose() const;

    // Scalar operations
    Matrix scalarMultiply(double scalar) const;
    Matrix scalarAdd(double scalar) const;

    // For debugging
    void print() const;

private:
    Matrix(const arma::mat& internal_matrix);
};

#endif // MATRIX_H