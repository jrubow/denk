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
    this->lossFunc = MSE();
}

BNN::BNN(std::vector<Layer> layers, uint64_t epochs, double learningRate, double trainingSplit, std::vector<Matrix> X, std::vector<Matrix> Y) {
    this->layers = layers;
    this->epochs = epochs;
    this->learningRate = learningRate;
    this->trainingSplit = trainingSplit;
    this->X = X;
    this->Y = Y;
    this->lossFunc = MSE();
}

double BNN::computeAverageLoss() {
    double totalLoss = 0.0;
    for (int64_t i = 0; i < X.size(); i++) {
        Matrix xInput = X[i];
        Matrix yExpected = Y[i];

        // Forward Propagation
        for (const Layer &layer : layers) {
            xInput = layer.forward(xInput);
        }

        // Compute Loss
        Matrix loss = lossFunc.compute(xInput, yExpected);
        totalLoss += loss.get(0, 0); // Assuming loss is a single value matrix
    }
    return totalLoss / X.size();
}

int64_t BNN::train(std::vector<Matrix> X, std::vector<Matrix> Y) {
    for (int i = 0; i < epochs; i++) {
        for (int64_t j = 0; j < X.size(); j++) {
            Matrix xInput = X[j];
            Matrix yExpected = Y[j];

            // Forward Propagation
            for (const Layer &layer : layers) {
                xInput = layer.forward(xInput);
            }

            // Compute Loss
            Matrix loss = lossFunc.compute(xInput, yExpected);
            printf("[Epoch %d, Sample %d] Average Loss: %.2f\n", i, j, computeAverageLoss());

            for (const Layer &layer: layers) {
                layer.backpropogate(loss);
                
            }
        }
    }


    return -1;
}