/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Logger source file.
 * Supports logging functionality
 */

#include <stdarg.h>
#include <stdio.h>
#include "logger.h"

void logm(char *msg) {
    printf("%s", msg);
}

void logf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

// Neural Network Logger Functions