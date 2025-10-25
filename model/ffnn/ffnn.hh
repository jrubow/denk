/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Feed Forward Neural Network (FFNN) class header file.
 * Supports ...
 */

#ifndef FFNN_H
#define FFNN_H

#include "core.hh"

class FFNN {
    public:
        // Training Parameters
        double trainingSplit;

        // Hyperparameters
        uint64_t epochs;
        double learningRate;
        Loss &loss;

        // Network Layers
        std::vector<Layer> layers;

        // Data
        std::vector<Matrix> *input;
        std::vector<Matrix> *expected;

        // Optimizer
        Optimizer &optimizer;

        // Advanced Configuration
        bool clippingEnabled = false;
        double clipMin = -1.0;
        double clipMax = 1.0;
        bool miniBatchEnabled = false;
        size_t batchSize = 32;
        bool momentumEnabled = false;
        double momentumFactor = 0.9;

        // Constructors
        FFNN(std::vector<Layer> layers, uint64_t epochs, double learningRate, double trainingSplit, Loss &loss, Optimizer &optimizer);
        FFNN(const FFNN&) = delete;
        FFNN& operator=(const FFNN&) = delete;
        FFNN(FFNN&&) noexcept = default;
        FFNN& operator=(FFNN&&) noexcept = default;
        
        // Training & Prediction
        status_t train(std::vector<Matrix> *xInput, std::vector<Matrix> *yExpected);
        status_t test();
        Matrix predict(Matrix input);
        double computeLoss(int index);
        double computeAverageTrainingLoss();
        double computeAverageTestLoss();
        const std::vector<Matrix> computeGradients(int64_t epoch);

        // Advanced Configuration
        void enableClipping(double minValue, double maxValue);
        void enableMiniBatch(size_t batchSize);
        void enableMomentum(double momentumFactor);
};

#endif // FFNN_H