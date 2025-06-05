/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief 
 * Implementation for the matrix class.
 * Supports element-wise and matrix multiplication, addition, and transposition.
 */



#include <armadillo>
#include <stdexcept>
#include "matrix.hh"

// Constructors
Matrix::Matrix(int r, int c) {
    // Verify r and c
    if (r < 1 || c < 1) {
        throw std::invalid_argument("Invalid argument row or col less than 1.");
    }
    rows = r;
    cols = c;
}

Matrix::Matrix(int r, int c, const std::vector<double>& initial_data) {
    // Verify r and c
    if (r < 1 || c < 1) {
        throw std::invalid_argument("Invalid argument row or col less than 1.");
    }

    // Verify integrity of initial_data
    int size = r * c;
    if (initial_data.empty()) {
        std::cerr << "WARNING: inital_data is empty. Program will proceed with empty data." << std::endl;
        for (int i = 0; i < size; i++) {
            data.push_back(0);
        }
    } else if (initial_data.size() > size) {
        std::cerr << "WARNING: inital_data is larger than given row & col params allow. Program will truncate data." << std::endl;
        data = std::vector<double>(initial_data.begin(), initial_data.begin() + size);
    } else if (initial_data.size() < size) {
        std::cerr << "WARNING: inital_data is smaller than given row & col params specificy. Program will extend data with 0.0." << std::endl;
        data = initial_data;
        for (int i = 0; i < size - initial_data.size(); i++) {
            data.push_back(0);
        }
    } else {
        data = initial_data;
    }
}


