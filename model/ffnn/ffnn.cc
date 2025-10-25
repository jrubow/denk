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

FFNN::FFNN(std::vector<Layer> layers, uint64_t epochs, double learningRate, double trainingSplit, Loss &lossRef, Optimizer &optimizerRef) :
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

double FFNN::computeAverageTrainingLoss() {
    double totalLoss = 0.0;
    for (size_t i = 0; i < (*input).size(); i++) {
        // use a local copy so the stored dataset element isn't mutated or
        // replaced by a pointer to layer internals
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
    return totalLoss / (*input).size();
}

double FFNN::computeAverageTestLoss() {
    double totalLoss = 0.0;
    int32_t start  = static_cast<int>(std::round(input->size() * trainingSplit));
    for (int32_t i = start; i < input->size(); i++) {
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
    return totalLoss / (*input).size();
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
    totalLoss += lossm.get(0, 0);

    return totalLoss / (*input).size();
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



status_t FFNN::train(std::vector<Matrix> *xInput, std::vector<Matrix> *yExpected) {
    this->input = xInput;
    this->expected = yExpected;

    int32_t end  = static_cast<int>(std::round(input->size() * trainingSplit));
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
            // _logf("\n\n");
            // for (int n = 0; n < layers.size(); n++) {
            //     for (int i = 0; i < layers[n].weights.getRows(); i++) {
            //         for (int j = 0; j < layers[n].weights.getCols(); j++) {
            //             _logf("%f ", layers[n].weights.get(i, j));
            //         }
            //         _logf("\n");
            //     }
            // }
            optimizer.updateParameters(layers, gradients, learningRate);
        }
        _logf("[EPOCH %d] Avg Loss: %.10f\n", (i + 1), computeAverageTrainingLoss());
        //printWeights();
    }
    return SUCCESS;
}

status_t FFNN::test() {
    _logf("\n[---------- START TEST ----------]\n");
    _logf("[TEST] Total Avg Loss: %.2f\n", computeAverageTestLoss());
    _logf("[---------- END TRAIN ----------]\n");
    return SUCCESS;
}