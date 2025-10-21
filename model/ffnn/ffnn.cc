/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Feed Forward Neural Network (FFNN) class implementation
 * 
 */

#include "core.hh"
#include "ffnn.hh"

FFNN::FFNN(std::vector<double> shape, uint64_t epochs, double learningRate, double trainingSplit, Loss &lossRef, Optimizer &optimizerRef) :
            layers(std::move(layers)),
            optimizer(optimizerRef),
            loss(lossRef) {
    this->epochs = epochs;
    this->learningRate = learningRate;
    this->trainingSplit = trainingSplit;
}

std::vector<Matrix> getWeights(const std::vector<Layer>& layers) {
    std::vector<Matrix> weights;
    weights.reserve(layers.size());

    for (const Layer& layer : layers) {
        weights.push_back(layer.weights);
    }

    return weights;
}

double FFNN::computeAverageLoss() {
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

double FFNN::computeLoss(int index) {
    double totalLoss = 0.0;
    Matrix xInput = input[index];
    Matrix yExpected = expected[index];

    for (Layer &layer : layers) {
        xInput = layer.forward(xInput);
    }

    // Compute Loss
    Matrix lossm = loss.compute(xInput, yExpected);
    totalLoss += lossm.get(0, 0);
    
    return totalLoss / input.size();
}

const std::vector<Matrix> FFNN::computeGradients(int64_t index) {

}

status_t FFNN::train() {
    for (int i = 0; i < epochs; i++) {
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

    return SUCCESS;
}

status_t FFNN::test() {
    _logf("\n[---------- START TEST ----------]\n");
    _logf("[TEST] Total Avg Loss: %.2f\n", computeAverageLoss());
    _logf("[---------- END TRAIN ----------]\n");
    return SUCCESS;
}