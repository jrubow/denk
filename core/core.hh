#ifndef CORE_H
#define CORE_H

#include "matrix/matrix.hh"

// Activators
#include "activator/activator.hh"
#include "activator/sigmoid.hh"

// Loss
#include "loss/loss.hh"
#include "loss/mse.hh"

// Layers
#include "layer/layer.hh"

// Optimizer
#include "optimizer/optimizer.hh"
#include "optimizer/sgd.hh"
#include "optimizer/newtonsmethod.hh"

// Logger
#include "logger/logger.h"

#endif // CORE_H