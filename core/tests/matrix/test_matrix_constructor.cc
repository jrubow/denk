/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * This file contains the tests for the matrix constructor
 */

#include "gtest/gtest.h"
#include "../../matrix/matrix.hh"
#include <vector>
#include <stdexcept>

TEST(MatrixConstructor, HandlesNegativeDimensions) {
    std::vector<double> empty_data;
    // EXPECT_THROW checks if the statement throws an exception of the specified type.
    EXPECT_THROW(Matrix m(-1, 5, empty_data), std::invalid_argument);
    EXPECT_THROW(Matrix m(5, -1, empty_data), std::invalid_argument);
    EXPECT_THROW(Matrix m(-2, -3, empty_data), std::invalid_argument);
}