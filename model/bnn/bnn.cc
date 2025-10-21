/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Basic Neural Network (BNN) class for managing a simple feedforward neural network
 * 
 */

#include "core.hh"
#include "bnn.hh"

BNN::BNN(std::vector<Layer> layers, uint64_t epochs, double learningRate, double trainingSplit, Loss &lossRef, Optimizer &optimizerRef) :
        layers(std::move(layers)),
        optimizer(optimizerRef),
        loss(lossRef) {
    this->epochs = epochs;
    this->learningRate = learningRate;
    this->trainingSplit = trainingSplit;
}

BNN::BNN(std::vector<Layer> layers, uint64_t epochs, double learningRate, double trainingSplit,
         std::vector<Matrix> input, std::vector<Matrix> expected, Loss &lossRef, Optimizer &optimizerRef) :
        layers(std::move(layers)),
        optimizer(optimizerRef),
        loss(lossRef) {
    this->epochs = epochs;
    this->learningRate = learningRate;
    this->trainingSplit = trainingSplit;
    this->input = input;
    this->expected = expected;
}

std::vector<Matrix> getWeights(const std::vector<Layer>& layers) {
    std::vector<Matrix> weights;
    weights.reserve(layers.size());

    for (const Layer& layer : layers) {
        weights.push_back(layer.weights);
    }

    return weights;
}

void BNN::printWeights() {
    _logf("\n--- BNN WEIGHTS ---\n");
    for (size_t i = 0; i < layers.size(); ++i) {
        Matrix weights = layers[i].weights;
        size_t rows = weights.getRows();
        size_t cols = weights.getCols();
        
        _logf("Layer %zu Weights (W%zu): %zu rows x %zu cols\n", i + 1, i + 1, rows, cols);
        
        // Iterate over rows
        for (size_t r = 0; r < rows; ++r) {
            _logf("  Row %zu: [", r);
            
            // Iterate over columns in the current row
            for (size_t c = 0; c < cols; ++c) {
                // Print the element value with precision for clarity
                _logf("%.6f", weights.get(r, c));
                
                // Add comma unless it's the last column
                if (c < cols - 1) {
                    _logf(", ");
                }
            }
            _logf("]\n");
        }
    }
    _logf("-------------------\n");
}


double BNN::computeAverageLoss() {
    double totalLoss = 0.0;
    for (size_t i = 0; i < input.size(); i++) {
        Matrix xInput = input[i];
        Matrix yExpected = expected[i];

        // Forward Propagation
        for (Layer &layer : layers) {
            xInput = layer.forward(xInput);
        }

        // Compute Loss
        Matrix lossm = loss.compute(layers[layers.size() - 1].neurons, yExpected);
        totalLoss += lossm.get(0, 0);
    }
    return totalLoss / input.size();
}

double BNN::computeLoss(int index) {
    double totalLoss = 0.0;
    Matrix xInput = input[index];
    Matrix yExpected = expected[index];

    // Forward Propagation
    for (Layer &layer : layers) {
        xInput = layer.forward(xInput);
    }

    // Compute Loss
    Matrix lossm = loss.compute(xInput, yExpected);
    totalLoss += lossm.get(0, 0);
    
    return totalLoss / input.size();
}

const std::vector<Matrix> BNN::computeGradients(int64_t index) {
    std::vector<Matrix> gradients(layers.size());
    std::vector<Matrix> deltas(layers.size());
    
    size_t n = layers.size();
    
    Matrix loss_deriv = loss.derivate(layers[n-1].neurons, expected[index]);
    Matrix activator_deriv = layers[n-1].activator.derivate(layers[n-1].neurons);
    
    deltas[n-1] = loss_deriv.multiplyElementwise(activator_deriv);

    // --- 2. Backpropagate Delta Through Layers ---
    
    for (size_t i = n - 2; i > 0; i--) {
        // Operation: (layers[i+1].weights.transpose() .multiply( deltas[i+1] ))
        
        Matrix transposed_weights = layers[i].weights;
        Matrix next_delta = deltas[i+1];

        // Check 2: Transposed Weights .multiply( Next Delta ) (Matrix Multiplication)
        if (transposed_weights.getCols() != next_delta.getRows()) {
            _logf("[ERROR] GRADIENT CHECK (Backprop %zu): Matrix multiplication mismatch!\n", i);
            _logf("  Weights[%zu] Transposed Dimensions: R=%zu, C=%zu\n", i + 1, transposed_weights.getRows(), transposed_weights.getCols());
            _logf("  Deltas[%zu] Dimensions: R=%zu, C=%zu\n", i + 1, next_delta.getRows(), next_delta.getCols());
            // You may want to return or throw here
        }
        Matrix multiplied_delta = transposed_weights.multiply(next_delta);

        // Check 3: Multiplied Delta .multiplyElementwise( Current Activator Deriv )
        // Both sides must have identical dimensions for element-wise multiplication.
        activator_deriv = layers[i].activator.derivate(layers[i].neurons);
        
        if (multiplied_delta.getRows() != activator_deriv.getRows() ||
            multiplied_delta.getCols() != activator_deriv.getCols()) {
            _logf("[ERROR] GRADIENT CHECK (Backprop %zu): Element-wise mismatch!\n", i);
            _logf("  Multiplied Delta Dimensions: R=%zu, C=%zu\n", multiplied_delta.getRows(), multiplied_delta.getCols());
            _logf("  Activator Deriv Dimensions: R=%zu, C=%zu\n", activator_deriv.getRows(), activator_deriv.getCols());
            // You may want to return or throw here
        }
        deltas[i] = multiplied_delta.multiplyElementwise(activator_deriv);
    }

    // --- 3. Compute Weight Gradients ---
    
    for (size_t i = 0; i < n - 1; i++) {
        // Operation: deltas[i] .multiply( layers[i].neurons.transpose() )
        Matrix current_delta = deltas[i + 1];
        Matrix transposed_neurons = layers[i].neurons.transpose();
        
        // Check 4: Delta .multiply( Transposed Neurons ) (Matrix Multiplication)
        if (current_delta.getCols() != transposed_neurons.getRows()) {
            _logf("[ERROR] GRADIENT CHECK (Gradient %zu): Matrix multiplication mismatch!\n", i);
            _logf("  Deltas[%zu] Dimensions: R=%zu, C=%zu\n", i, current_delta.getRows(), current_delta.getCols());
            _logf("  Neurons[%zu] Transposed Dimensions: R=%zu, C=%zu\n", i, transposed_neurons.getRows(), transposed_neurons.getCols());
            // You may want to return or throw here
        }

        gradients[i] = current_delta.multiply(transposed_neurons);
    }

    return gradients;
}

// TODO: implement mini-batch and batch gd
status_t BNN::train() {
    for (int i = 0; i < epochs; i++) {
        _logf("\n[---------- START TRAIN ----------]\n", (i + 1), epochs);
        for (size_t j = 0; j < input.size(); j++) {
            Matrix xInput = input[j];
            Matrix yExpected = expected[j];
            // Forward Propagation
            for (Layer &layer : layers) {
                xInput = layer.forward(xInput);
            }
            

            

            // Update weights
            std::vector<Matrix> gradients = computeGradients(j);
            optimizer.updateParameters(layers, gradients, learningRate);

        }
        _logf("[EPOCH %d] Avg Loss: %.10f\n", (i + 1), computeAverageLoss());
        //printWeights();

    }

    _logf("[---------- END TRAIN ----------]\n");

    return SUCCESS;
}

status_t BNN::test() {
    _logf("\n[---------- START TEST ----------]\n");
    _logf("[TEST] Total Avg Loss: %.2f\n", computeAverageLoss());
    _logf("[---------- END TRAIN ----------]\n");
    return SUCCESS;
}