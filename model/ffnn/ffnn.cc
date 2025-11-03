/**
 * @file
 * @copyright Copyright (c) Jo h Rubow (jrubow). All rights reserved.
 *
 * @brief
 * Feed Forward Neural Network (FFNN) class implementation
 * 
 */

#include "core.hh"
#include "ffnn.hh"

FFNN::FFNN(std::vector<Layer> layers, uint64_t epochs, double trainingSplit, Loss &lossRef, Optimizer &optimizerRef) :
            layers(std::move(layers)),
            optimizer(optimizerRef),
            loss(lossRef) {
    this->epochs = epochs;
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

double FFNN::computeAverageTrainingLoss() {
    double totalLoss = 0.0;
    int32_t end  = static_cast<int>(std::round(input->size() * trainingSplit));
    if (end <= 0) return 0.0;
    for (int32_t i = 0; i < end; i++) {
        Matrix activation = (*input)[i];
        Matrix *yExpected = &(*expected)[i];

        // Forward Propagation
        for (Layer &layer : layers) {
            Matrix modified = activation.appendRow(1.0);
            Matrix *out = layer.forward(&modified);
            activation = *out;
        }

        // Compute Loss using the final layer's neurons
        Matrix lossm = loss.compute(&layers[layers.size() - 1].neurons, yExpected);
        totalLoss += lossm.get(0, 0);
    }
    return totalLoss / static_cast<double>(end);
}

double FFNN::computeAverageTestLoss() {
    double totalLoss = 0.0;
    int32_t start  = static_cast<int>(std::round(input->size() * trainingSplit));
    int32_t inputSize = static_cast<int32_t>(input->size());
    if (start >= inputSize) return 0.0;
    for (int32_t i = start; i < inputSize; i++) {
        Matrix activation = (*input)[i];
        Matrix *yExpected = &(*expected)[i];

        // Forward Propagation
        for (Layer &layer : layers) {
            Matrix modified = activation.appendRow(1.0);
            Matrix *out = layer.forward(&modified);
            activation = *out;
        }

        // Compute Loss
        Matrix lossm = loss.compute(&layers[layers.size() - 1].neurons, yExpected);
        totalLoss += lossm.get(0, 0);
    }
    int32_t testCount = inputSize - start;
    return totalLoss / static_cast<double>(testCount);
}

double FFNN::computeLoss(int index) {
    double totalLoss = 0.0;
    Matrix activation = (*input)[index];
    Matrix *yExpected = &(*expected)[index];

    for (Layer &layer : layers) {
        Matrix modified = activation.appendRow(1.0);
        Matrix *out = layer.forward(&modified);
        activation = *out;
    }

    // Compute Loss
    Matrix lossm = loss.compute(&layers[layers.size() - 1].neurons, yExpected);
    return lossm.get(0, 0);
}

const std::vector<Matrix> FFNN::computeGradients(int64_t index) {
    const uint32_t totalLayers = layers.size();
    std::vector<Matrix> deltas(totalLayers);
    Matrix lossDeriv = loss.derivate(&layers[totalLayers - 1].neurons, &(*expected)[index]);
    std::vector<Matrix> gradients(totalLayers);
    
    // Output Layer
    // TODO: simplify or move to layers class for dz. convert to pointers
    deltas[totalLayers - 1] = std::move(lossDeriv.multiplyElementwise(layers[totalLayers - 1].activator.derivate(layers[totalLayers - 1].preActivation)));

    // Hidden Layers
    for (int i = totalLayers - 2; i >= 0; i--) {
    deltas[i] = std::move(layers[i+1].weights.transpose().multiply(deltas[i + 1]).removeLastRow()
            .multiplyElementwise(layers[i].activator.derivate(layers[i].preActivation)));
    }

    // Compute Gradients
    for (int i = 1; i < totalLayers; i++) {
        gradients[i] = deltas[i].multiply(layers[i-1].neurons.appendRow(1.0).transpose());
    }
    
    gradients[0] = deltas[0].multiply((*input)[index].appendRow(1.0).transpose());

    // for (int i = 0; i < totalLayers; i++) {
    //     _logf("Gradient for layer %d: \n", i);
    //     for(int j = 0; j < gradients[i].getRows(); j++) {
    //         for (int c = 0; c < gradients[i].getCols(); c++) {
    //             _logf("%f ", gradients[i].get(j, c));
    //         }
    //         _logf("\n");
    //     }
    // }

    return gradients;
}

void FFNN::printWeights() {
    _logf("\n--- BNN WEIGHTS ---\n");
    for (size_t i = 0; i < layers.size(); ++i) {
        Matrix weights = layers[i].weights;
        size_t rows = weights.getRows();
        size_t cols = weights.getCols();
        
        _logf("Layer %zu Weights (W%zu): %zu rows x %zu cols\n", i + 1, i + 1, rows, cols);
        
        for (size_t r = 0; r < rows; ++r) {
            _logf("  Row %zu: [", r);
            for (size_t c = 0; c < cols; ++c) {
                _logf("%.6f", weights.get(r, c));
                if (c < cols - 1) {
                    _logf(", ");
                }
            }
            _logf("]\n");
        }
    }
    _logf("-------------------\n");
}



status_t FFNN::train(std::vector<Matrix> *xInput, std::vector<Matrix> *yExpected) {
    this->input = xInput;
    this->expected = yExpected;

    int32_t end  = static_cast<int>(std::round(input->size() * trainingSplit));

    printWeights();
    for (int i = 0; i < epochs; i++) {
        for (size_t j = 0; j < end; j++) {
            Matrix activation = (*input)[j];
            Matrix *yExpected = &(*expected)[j];

            for (Layer &layer : layers) {
                Matrix modified = activation.appendRow(1.0);
                Matrix *out = layer.forward(&modified);
                activation = *out;
            }

            // Update weights
            std::vector<Matrix> gradients = computeGradients(j);
            optimizer.updateParameters(layers, gradients);
        }
        
        _logf("[EPOCH %d] Avg Loss: %.10f\n", (i + 1), computeAverageTrainingLoss());
        // printWeights();
        
    }
    return SUCCESS;
}

status_t FFNN::test() {
    _logf("\n[---------- START TEST ----------]\n");
    _logf("[TEST] Total Avg Loss: %.2f\n", computeAverageTestLoss());
    _logf("[---------- END TRAIN ----------]\n");
    return SUCCESS;
}