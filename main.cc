#include <iostream>
#include <fcntl.h>
#include <random>
#include <cmath>

#include "core.hh"
#include "model.hh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void printTitle() {
    #ifdef _WIN32
        std::cout <<
    "       ___           ___           ___           ___     \n"
    "      /\\  \\         /\\  \\         /\\__\\         /\\__\\    \n"
    "    /::\\  \\       /::\\  \\       /::|  |       /:/  /    \n"
    "   /:/\\:\\  \\     /:/\\:\\  \\     /:|:|  |      /:/__/     \n"
    "  /:/  \\:\\__\\   /::\\~\\:\\  \\   /:/|:|  |__   /::\\__\\____ \n"
    " /:/__/ \\:|__| /:/\\:\\ \\:\\__\\ /:/ |:| /\\__\\ /:/\\:::::\\__\\\n"
    " \\:\\  \\ /:/  / \\:\\~\\:\\ \\/__/ \\/__|:|/:/  / \\/_|:|~~|~   \n"
    "  \\:\\  /:/  /   \\:\\ \\:\\__\\       |:/:/  /     |:|  |    \n"
    "   \\:\\/:/  /     \\:\\ \\/__/       |::/  /      |:|  |    \n"
    "    \\::/__/       \\:\\__\\         /:/  /       |:|  |    \n"
    "     --            \\/__/         \\/__/         \\|__|   \n"
    "                     Joshua James\n\n";
    #else
        // Linux / macOS
        std::cout << "\n██████╗ ███████╗███╗   ██╗██╗  ██╗\n"
                       "██╔══██╗██╔════╝████╗  ██║██║ ██╔╝\n"
                       "██╔═███║█████╗  ██╔██╗ ██║█████╔╝ \n"
                       "██║  ██║██╔══╝  ██║╚██╗██║██╔═██╗ \n"
                       "██████╔╝███████╗██║ ╚████║██║  ██╗\n"
                       "╠═╣═══╝ ╚══════╝╚═╝  ╚═══╝╚═╝  ╠═╣\n"
                       "╠═╣  Machine Learning Library  ╠═╣\n"
                       "╚═╩═══╡   Joshua James   ╞═════╩═╝\n\n";
    #endif
}

void ffnn0() {
    printf("\n[Starting FFNN Default 1: Learn AND Gate ]\n");
    // Create Activators, Loss, and Optimzier
    Sigmoid sigmoid;
    Identity identity;
    MSE loss;
    BCE bce_loss;
    SGD optimizer(0.05);


    // Create Layers
    std::vector<Layer> layers;
    layers.push_back(Layer(2, identity, 2, 0));
    layers.push_back(Layer(2, sigmoid, 2, 0));
    layers.push_back(Layer(2, sigmoid, 1, 0));
    

    // Data Setup
    std::vector<double> input1 = {1.0, 1.0};
    std::vector<double> input2 = {0.0, 1.0};
    std::vector<double> input3 = {1.0, 0.0};
    std::vector<double> input4 = {0.0, 0.0};

    std::vector<double> outputTrue = {1.0};
    std::vector<double> outputFalse = {0.0};

    std::vector<Matrix> input;
    std::vector<Matrix> output;
    
    input.push_back(Matrix(2, 1, input1));
    input.push_back(Matrix(2, 1, input2));
    input.push_back(Matrix(2, 1, input3));
    input.push_back(Matrix(2, 1, input4));

    output.push_back(Matrix(1, 1, outputTrue));
    output.push_back(Matrix(1, 1, outputFalse));
    output.push_back(Matrix(1, 1, outputFalse));
    output.push_back(Matrix(1, 1, outputFalse));
    

    
    // Model Setup: use BCE with sigmoid output and increase learning rate to 0.05
    FFNN model(std::move(layers), 100000, 1, bce_loss, optimizer);

    // Training
    model.train(&input, &output);

    // Testing
    model.test();

    printf("\n[Finished FFNN Test]\n");
}

void ffnn1() {
    printf("\n[Starting FFNN Default 1: Learn AND Gate ]\n");
    // Create Activators, Loss, and Optimzier
    Sigmoid sigmoid;
    Identity identity;
    MSE loss;
    BCE bce_loss;
    Momentum optimizer(0.05, 0.9, {{2,3}, {2,3}, {1,3}});

    // Create Layers
    std::vector<Layer> layers;
    layers.push_back(Layer(2, identity, 2, 0));
    layers.push_back(Layer(2, sigmoid, 2, 0));
    layers.push_back(Layer(2, sigmoid, 1, 0));
    

    // Data Setup
    std::vector<double> input1 = {1.0, 1.0};
    std::vector<double> input2 = {0.0, 1.0};
    std::vector<double> input3 = {1.0, 0.0};
    std::vector<double> input4 = {0.0, 0.0};

    std::vector<double> outputTrue = {1.0};
    std::vector<double> outputFalse = {0.0};

    std::vector<Matrix> input;
    std::vector<Matrix> output;
    
    input.push_back(Matrix(2, 1, input1));
    input.push_back(Matrix(2, 1, input2));
    input.push_back(Matrix(2, 1, input3));
    input.push_back(Matrix(2, 1, input4));

    output.push_back(Matrix(1, 1, outputTrue));
    output.push_back(Matrix(1, 1, outputFalse));
    output.push_back(Matrix(1, 1, outputFalse));
    output.push_back(Matrix(1, 1, outputFalse));
    

    
    // Model Setup: use BCE with sigmoid output and increase learning rate to 0.05
    FFNN model(std::move(layers), 100000, 1, bce_loss, optimizer);

    // Training
    model.train(&input, &output);

    // Testing
    model.test();

    printf("\n[Finished FFNN Test]\n");
}

void ffnn2() {
    printf("\n[Starting FFNN Regression Test: Learn exp(x) on [-1,1]]\n");

    // Activators, loss, optimizer
    Sigmoid sigmoid2;
    Identity identity;
    MSE loss;
    SGD optimizer(0.01);

    // Create Layers for a small regression network: 1 -> 10 -> 10 -> 1
    std::vector<Layer> layers;
    layers.push_back(Layer(1, identity, 10, 0));   // input dim 1 -> 10 neurons
    layers.push_back(Layer(10, sigmoid2, 10, 0));  // hidden
    layers.push_back(Layer(10, identity, 1, 0));   // output linear

    // Create dataset: sample exp(x) on [-1, 1]
    const int N = 200;
    std::vector<Matrix> input;
    std::vector<Matrix> output;
    input.reserve(N);
    output.reserve(N);

    for (int i = 0; i < N; ++i) {
        double x = -1.0 + 2.0 * i / static_cast<double>(N - 1);
        double y = std::exp(x) + 2 * x;
        input.push_back(Matrix(1, 1, std::vector<double>{x}));
        output.push_back(Matrix(1, 1, std::vector<double>{y}));
    }

    // Model: use MSE loss for regression
    FFNN model(std::move(layers), 5000, 0.8, loss, optimizer);

    model.train(&input, &output);

    // Quick test: print some predictions vs actual
    printf("\nSample predictions after training:\n");
    for (int i = 0; i < 10; ++i) {
        int idx = i * (N / 10);
        Matrix activation = input[idx];
        for (Layer &layer : model.layers) {
            Matrix modified = activation.appendRow(1.0);
            Matrix *out = layer.forward(&modified);
            activation = *out;
        }
        double pred = model.layers.back().neurons.get(0, 0);
        double actual = output[idx].get(0, 0);
        printf("x=%.4f pred=%.6f actual=%.6f\n", input[idx].get(0,0), pred, actual);
    }

    printf("[Finished FFNN Regression Test]\n");
}
void ffnn3() {
    printf("\n[Starting FFNN Complex Test: Learn Rosenbrock function f(x,y) = (1-x)² + 100(y-x²)²]\n");

    // Activators, loss, optimizer
    Tanh tanh;
    Sigmoid sigmoid;
    Identity identity;
    ReLu relu;
    MSE loss;
    Momentum optimizer(0.001, 0.9, {{16, 3}, {32,17}, {32,33}, {16,33}, {1,17}});
    // SGD optimizer(0.001);
    
    std::vector<Layer> layers;
    layers.push_back(Layer(2, identity, 16, 2));
    layers.push_back(Layer(16, relu, 32, 2));
    layers.push_back(Layer(32, relu, 32, 2));
    layers.push_back(Layer(32, relu, 16, 2));
    layers.push_back(Layer(16, identity, 1, 2));

    const int N = 40; 
    std::vector<Matrix> input;
    std::vector<Matrix> output;
    input.reserve(N * N);
    output.reserve(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = -2.0 + 4.0 * i / static_cast<double>(N - 1);
            double y = -2.0 + 4.0 * j / static_cast<double>(N - 1);
            
            double z = std::pow(1.0 - x, 2.0) + 100.0 * std::pow(y - x*x, 2.0);
            z = std::log1p(z);
            input.push_back(Matrix(2, 1, std::vector<double>{x, y}));
            output.push_back(Matrix(1, 1, std::vector<double>{z}));
        }
    }

    // Model: smaller learning rate due to function complexity
    FFNN model(std::move(layers), 10000, 0.8, loss, optimizer);
    model.train(&input, &output);

    // Test predictions on a few key points
    printf("\nSample predictions after training:\n");
    std::vector<std::pair<double, double>> test_points = {
        {1.0, 1.0},
        {0.0, 0.0},
        {2.0, 4.0},
        {-1.0, 1.0}
    };

    for (const auto& point : test_points) {
        double x = point.first;
        double y = point.second;
        Matrix test_input(2, 1, std::vector<double>{x, y});
        
        Matrix activation = test_input;
        for (Layer &layer : model.layers) {
            Matrix modified = activation.appendRow(1.0);
            Matrix *out = layer.forward(&modified);
            activation = *out;
        }
        
        double actual = std::pow(1.0 - x, 2.0) + 100.0 * std::pow(y - x*x, 2.0);
        double pred_log = model.layers.back().neurons.get(0, 0);
        double pred = std::exp(pred_log) - 1.0;

        printf("(x,y)=(%.2f,%.2f) pred=%.6f actual=%.6f\n", x, y, pred, actual);
    }

    printf("\n[100 Random Sample Predictions in Range (-2, 2) x (-2, 2)]\n");
    for (int k = 0; k < 100; ++k) {
        double x = -2.0 + 4.0 * (rand() / static_cast<double>(RAND_MAX));
        double y = -2.0 + 4.0 * (rand() / static_cast<double>(RAND_MAX));
        Matrix test_input(2, 1, std::vector<double>{x, y});

        Matrix activation = test_input;
        for (Layer &layer : model.layers) {
            Matrix modified = activation.appendRow(1.0);
            Matrix *out = layer.forward(&modified);
            activation = *out;
        }

        double actual = std::pow(1.0 - x, 2.0) + 100.0 * std::pow(y - x*x, 2.0);
        double pred_log = model.layers.back().neurons.get(0, 0);
        double pred = std::exp(pred_log) - 1.0;

        printf("(x,y)=(%.3f,%.3f) pred=%.6f actual=%.6f\n", x, y, pred, actual);
    }

    printf("[Finished FFNN Complex Test]\n");
}


int main(int argc, int *argv[]) {
    printTitle();
    
    uint16_t shell = 1;
    char input[100];
    while (shell) {
        printf("[cmd]> ");
        scanf("%99s", input);

        if (strcmp(input, "exit") == 0) {
            shell = 0;
        }  else if (strcmp(input, "ffnn0") == 0) {
            ffnn0();
        }  else if (strcmp(input, "ffnn1") == 0) {
            ffnn1();
        } else if (strcmp(input, "ffnn2") == 0) {
            ffnn2();
        } else if (strcmp(input, "ffnn3") == 0) {
            ffnn3();
        }
    }

    return 0;
}
