/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief 
 * Implementation for the Matrix class.
 * Supports element-wise and matrix multiplication, addition, and transposition.
 */

#include "matrix.hh"
#include <stdexcept>
#include <iostream>
#include <armadillo>

// Constructors

// Public constructor to create a matrix of a given size, initialized to zeros.
Matrix::Matrix(int r, int c) {
    if (r < 0 || c < 0) {
        throw std::invalid_argument("Invalid argument: row or col cannot be negative.");
    }
    this->data.zeros(r, c);
}

Matrix::Matrix(int r, int c, int init) {
    if (r < 0 || c < 0) {
        throw std::invalid_argument("Invalid argument: row or col cannot be negative.");
    }
    this->data.set_size(r, c);
    this->data.fill(init);
}

// Public constructor to create a matrix from an existing std::vector.
Matrix::Matrix(int r, int c, const std::vector<double>& initial_data) {
    // Verify r and c
    if (r < 0 || c < 0) {
        throw std::invalid_argument("[ERROR]: Invalid argument: row or col cannot be negative.");
    }

    size_t expected_size = static_cast<size_t>(r) * static_cast<size_t>(c);

    // IMPORTANT: This constructor assumes the std::vector 'initial_data'
    // is already in COLUMN-MAJOR format to match Armadillo's standard.
    if (initial_data.size() == expected_size) {
        this->data = arma::mat(initial_data.data(), r, c);
    } else if (initial_data.empty()) {
        std::cerr << "[WARNING] initial_data is empty. Matrix will be initialized with zeros." << std::endl;
        this->data.zeros(r, c);
    } else if (initial_data.size() > expected_size) {
        std::cerr << "[WARNING] initial_data is larger than r*c. Data will be truncated." << std::endl;
        this->data = arma::mat(initial_data.data(), r, c);
    } else { // initial_data.size() < expected_size
        std::cerr << "[WARNING] initial_data is smaller than r*c. Matrix will be padded with 0.0." << std::endl;
        this->data.zeros(r, c);
        std::memcpy(this->data.memptr(), initial_data.data(), initial_data.size() * sizeof(double));
    }
}

// Private constructor - for fast implementation
Matrix::Matrix(const arma::mat& internal_matrix) : data(internal_matrix) {}


// Getters
int Matrix::getRows() const {
    return this->data.n_rows;
}

int Matrix::getCols() const {
    return this->data.n_cols;
}

double Matrix::get(int r, int c) const {
    return this->data(r, c);
}

// Setters
void Matrix::set(int r, int c, double value) {
    this->data(r, c) = value;
}

// Basic Operations
Matrix Matrix::add(const Matrix &mat) const {
    // Validate arguments, rows and cols must be equal
    if (this->getRows() != mat.getRows() || this->getCols() != mat.getCols()) {
        throw std::invalid_argument("[ERROR] Matrices must have the same dimensions for addition.");
    }

    arma::mat result = this->data + mat.data;
    return Matrix(result);
}

Matrix Matrix::subtract(const Matrix &mat) const {
    // Validate arguments, rows and cols must be equal
    if (this->getRows() != mat.getRows() || this->getCols() != mat.getCols()) {
        throw std::invalid_argument("[ERROR] Matrices must have the same dimesions for subtraction.");
    }

    arma::mat result = this->data - mat.data;
    return Matrix(result);
}

Matrix Matrix::multiplyElementwise(const Matrix &mat) const {
    if (this->getRows() != mat.getRows() || this->getCols() != mat.getCols()) {
        throw std::invalid_argument("[ERROR] Matrices must have the same dimensions for elementwise multiplication.");
    }

    arma::mat result = this->data % mat.data;
    return Matrix(result);
}

Matrix Matrix::multiply(const Matrix &mat) const {
    if (this->getCols() != mat.getRows()) {
        throw std::invalid_argument("[ERROR] Dimesnions do not match for matrix multiplication operation.");
    }

    arma::mat result = this->data * mat.data;
    return Matrix(result);
}

Matrix Matrix::toPower(double power) const {
    arma::mat result = arma::pow(data, power);
    return result;
}

Matrix Matrix::exponent() const {
    arma::mat result = arma::exp(data);
    return result;
}

// Scalar Operations
Matrix Matrix::scalarMultiply(double scalar) const {
    arma::mat result = scalar * this->data;
    return Matrix(result);
}

Matrix Matrix::scalarAdd(double scalar) const {
    arma::mat result = scalar + this->data;
    return Matrix(result);
}

// Other Operations
Matrix Matrix::transpose() const {
    arma::mat result = this->data.t();
    return Matrix(result);
}

// Helper functions
void Matrix::uRandomize(double scalar) {
    data = scalar * data.randu(data.n_cols, data.n_rows);
}