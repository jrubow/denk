/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Matrix class for basic linear algebra operations.
 * Supports element-wise and matrix multiplication, addition, and transposition.
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

class Activation {
public:
    Activation();
    double activate(double input);
};

#endif // ACTIVATION_H