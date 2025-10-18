/**
 * @file
 * @copyright Copyright (c) Josh Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Basic Neural Network (BNN) class header file.
 * Supports ...
 */

#ifndef BNN_H
#define BNN_H

#include "core.hh"

class BNN {
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
        std::vector<Matrix> input;
        std::vector<Matrix> expected;

        // Optimizer
        Optimizer &optimizer;

        // Constructors
        BNN(std::vector<Layer> layers, uint64_t epochs, double learningRate, double trainingSplit, Loss &loss, Optimizer &optimizer);
        BNN(std::vector<Layer> layers, uint64_t epochs, double learningRate, double trainingSplit, std::vector<Matrix> input, std::vector<Matrix> expected, Loss &loss, Optimizer &optimizer);
        
        // Training & Prediction
        status_t train();
        status_t test();
        Matrix predict(Matrix input);
        double computeLoss(int index);
        double computeAverageLoss();
        const std::vector<Matrix> computeGradients(int64_t epoch);
        void printWeights();


        // Getters & Setters
        void setData(std::vector<Matrix> X, std::vector<Matrix> Y);
        void setX(std::vector<Matrix> X);
        void setY(std::vector<Matrix> Y);
        std::vector<Matrix> getX() const;
        std::vector<Matrix> getY() const;

        void addLayer(Layer layer);
        void setLayers(std::vector<Layer> layers);
        std::vector<Layer> getLayers() const;

        void setEpochs(int64_t epochs);
        int64_t getEpochs() const;

        void setLearningRate(double learningRate);
        double getLearningRate() const; 

        void setTrainingSplit(double trainingSplit);
        double getTrainingSplit() const;
};

#endif // BNN_H