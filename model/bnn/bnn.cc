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

BNN::BNN(std::vector<Layer> layers, uint64_t epochs, double learningRate, double trainingSplit) {
    this->layers = layers;
    this->epochs = epochs;
    this->learningRate = learningRate;
    this->trainingSplit = trainingSplit;
    this->loss = MSE();
    this->optimizer = SGD();
}

BNN::BNN(std::vector<Layer> layers, uint64_t epochs, double learningRate, double trainingSplit, std::vector<Matrix> input, std::vector<Matrix> expected) {
    this->layers = layers;
    this->epochs = epochs;
    this->learningRate = learningRate;
    this->trainingSplit = trainingSplit;
    this->input = input;
    this->expected = expected;
    this->loss = MSE();
    this->optimizer = SGD();
}

std::vector<Matrix> getWeights(const std::vector<Layer>& layers) {
    std::vector<Matrix> weights;
    weights.reserve(layers.size());

    for (const Layer& layer : layers) {
        weights.push_back(layer.weights);
    }

    return weights;
}


double BNN::computeAverageLoss() {
    double totalLoss = 0.0;
    for (int64_t i = 0; i < input.size(); i++) {
        Matrix xInput = input[i];
        Matrix yExpected = expected[i];

        // Forward Propagation
        for (const Layer &layer : layers) {
            xInput = layer.forward(xInput);
        }

        // Compute Loss
        Matrix lossm = loss.compute(xInput, yExpected);
        totalLoss += lossm.get(0, 0);
    }
    return totalLoss / input.size();
}

std::vector<Matrix> BNN::computeGradients(int64_t index) {
    std::vector<Matrix> gradients(layers.size());
    std::vector<Matrix> deltas(layers.size());

    // Compute delta for output layer
    int n = layers.size();
    deltas[n-1] = loss.derivate(expected[index], layers[n-1].neurons)
                  .multiplyElementwise(layers[n-1].activator.derivate(layers[n-1].neurons));

    // Backpropagate delta through layers
    for (int i = n - 2; i >= 0; i--) {
        deltas[i] = (layers[i+1].weights.transpose().multiply(deltas[i+1]))
                     .multiplyElementwise(layers[i].activator.derivate(layers[i].neurons));
    }

    // Compute weight gradients
    for (int i = 0; i < n; i++) {
        gradients[i] = deltas[i].multiply(layers[i].neurons.transpose());
    }

    return gradients;
}


int16_t BNN::train() {
    for (int i = 0; i < epochs; i++) {
        for (int64_t j = 0; j < input.size(); j++) {
            Matrix xInput = input[j];
            Matrix yExpected = expected[j];

            // Forward Propagation
            for (const Layer &layer : layers) {
                xInput = layer.forward(xInput);
            }

            logf("[Epoch %d, Sample %d] Average Loss: %.2f\n", (i + 1), (j + 1), computeAverageLoss());

            // Update weights
            computeGradients(j);
            optimizer.updateParameters( gradients);

        }
    }


    return -1;
}