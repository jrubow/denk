#include <iostream>
#include <io.h>
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

void bnntest() {
    printf("\n[Starting BNN Test: AND Gate ]\n");
    // Create Activators, Loss, and Optimzier
    Activator &sigmoid = Sigmoid();
    Activator &identity = Identity();
    Loss &loss = MSE();
    Optimizer &optimizer = SGD();

    // Create Layers
    std::vector<Layer> layers;
    layers.push_back(Layer(3, identity, 2));
    layers.push_back(Layer(2, sigmoid, 1));
    layers.push_back(Layer(1, identity, 1));
    

    // Data Setup
    std::vector<double> input1 = {1.0, 1.0, 1.0};
    std::vector<double> input2 = {1.0, 0.0, 1.0};
    std::vector<double> input3 = {1.0, 1.0, 0.0};
    std::vector<double> input4 = {1.0, 0.0, 0.0};

    std::vector<double> outputTrue = {1.0};
    std::vector<double> outputFalse = {0.0};

    std::vector<Matrix> input;
    std::vector<Matrix> output;
    
    input.push_back(Matrix(3, 1, input1));
    input.push_back(Matrix(3, 1, input2));
    input.push_back(Matrix(3, 1, input3));
    input.push_back(Matrix(3, 1, input4));

    output.push_back(Matrix(1, 1, outputTrue));
    output.push_back(Matrix(1, 1, outputFalse));
    output.push_back(Matrix(1, 1, outputFalse));
    output.push_back(Matrix(1, 1, outputFalse));
    

    
    // Model Setup
    BNN model(std::move(layers), 100000, 0.001, 0.2, input, output, loss, optimizer);

    // Training
    model.train();

    // Testing
    model.test();

    printf("\n[Finished BNN Test]\n");
}

double zfunc(double theta, double r) {
    return std::exp(-0.25 * r * r) * (2 * std::sin(M_PI * r) - r * std::cos(3 * theta));
}

void bnntest2() {
    printf("\n[Starting BNN Test 2: AND Gate ]\n");
    // Create Activators, Loss, and Optimzier
    Activator &sigmoid = Sigmoid();
    Activator &identity = Identity();
    Loss &loss = MSE();
    Optimizer &optimizer = SGD();

    // Create Layers
    std::vector<Layer> layers;
    layers.push_back(Layer(2, identity, 32));
    layers.push_back(Layer(32, sigmoid, 64));
    layers.push_back(Layer(64, sigmoid, 32));
    layers.push_back(Layer(32, sigmoid, 1));
    layers.push_back(Layer(1, identity, 1));
    

    // Data Setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist_r(0.0, 4.0);
    std::uniform_real_distribution<double> dist_theta(0.0, 2 * M_PI);
    std::vector<Matrix> input;
    std::vector<Matrix> output;
    
    for (int i = 0; i < 5000; i++) {
        double r = dist_r(gen);
        double theta = dist_theta(gen);
        std::vector<double> newInput= {r, theta};
        std::vector<double> newOutput = {zfunc(theta, r)};
        input.push_back(Matrix(2, 1, newInput));
        output.push_back(Matrix(1, 1, newOutput));
    }
    

    
    // Model Setup
    BNN model(std::move(layers), 100000, 0.001, 0.2, input, output, loss, optimizer);

    // Training
    model.train();

    // Testing
    model.test();

    printf("\n[Finished BNN Test]\n");
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
        } else if (strcmp(input, "bnn") == 0) {
            bnntest();
        } else if (strcmp(input, "bnn2") == 0) {
            bnntest2();
        }
    }

    return 0;
}
