#ifndef CORE_H
#define CORE_H

#include "matrix/matrix.hh"

// Activators
#include "activator/activator.hh"
#include "activator/sigmoid.hh"
#include "activator/identity.hh"
#include "activator/tanh.hh"
#include "activator/relu.hh"
#include "activator/leakyrelu.hh"

// Loss
#include "loss/loss.hh"
#include "loss/mse.hh"
#include "loss/bce.hh"

// Layers
#include "layer/layer.hh"

// Optimizer
#include "optimizer/optimizer.hh"
#include "optimizer/sgd.hh"
#include "optimizer/newtonsmethod.hh"
#include "optimizer/momentum.hh"


// System
 
// Logger
#include "logger/logger.h"

// Status
#include "status/status.h"

#endif // CORE_H