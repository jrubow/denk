/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * This file contains the tests for the sigmoid activation class
 */

#include "gtest/gtest.h"
#include "../activator/sigmoid.cc"
#include <cmath>

TEST(SigmoidTests, InputOne) {
    Sigmoid a = Sigmoid();
    ASSERT_EQ(a.activate(1.0), (1.0 / (1.0 + std::exp(-1.0))));
}

TEST(SigmoidTests, InputTwo) {
    Sigmoid a = Sigmoid();
    ASSERT_EQ(a.activate(12345.6789), (1.0 / (1.0 + std::exp(-12345.6789))));
}

TEST(SigmoidTests, InputThree) {
    Sigmoid a = Sigmoid();
    ASSERT_EQ(a.activate(2.1435), (1.0 / (1.0 + std::exp(-2.1435))));
}
 