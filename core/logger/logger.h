/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Logger header file.
 * Support logging functionality
 */

#include <stdarg.h>
#include <stdio.h>

#ifndef LOGGER_H
#define LOGGER_H

#ifdef DISABLE_LOGGING

#define logm(msg)  ((void)0)
#define logf(format, ...) ((void)0)


#else

void logm(char *msg);

void logf(const char *format, ...);

#endif

#endif // LOGGER_H