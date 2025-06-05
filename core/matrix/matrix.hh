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

#include <vector>
#include <iostream>

class Matrix {
public:
    int rows;
    int cols;
    std::vector<double> data;

    // Constructors
    Matrix(int r, int c);
    Matrix(int r, int c, const std::vector<double>& initial_data);

    // Accessors
    double get(int r, int c) const;
    void set(int r, int c, double val);

    // Basic operations
    inline Matrix add(const Matrix& other) const; 
    inline Matrix subtract(const Matrix& other) const; // Element-wise subtraction
    inline Matrix multiply_elementwise(const Matrix& other) const; // Hadamard product
    inline Matrix multiply(const Matrix& other) const; // Matrix multiplication (dot product)
    inline Matrix transpose() const;

    // Scalar operations
    Matrix scalar_multiply(double scalar) const;
    Matrix scalar_add(double scalar) const;

    // For debugging
    void print() const;
};

#endif // MATRIX_H