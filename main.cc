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

void ffnntest() {
    printf("\n[Starting FFNN Default 1: Learn AND Gate ]\n");
    // Create Activators, Loss, and Optimzier
    Sigmoid sigmoid;
    Identity identity;
    MSE loss;
    SGD optimizer;


    // Create Layers
    std::vector<Layer> layers;
    layers.push_back(Layer(2, identity, 2));
    layers.push_back(Layer(2, sigmoid, 1));
    layers.push_back(Layer(1, identity, 1));
    

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
    

    
    // Model Setup
    FFNN model(std::move(layers), 100000, 0.1, 1, loss, optimizer);

    // Training
    model.train(&input, &output);

    // Testing
    model.test();

    printf("\n[Finished FFNN Test]\n");
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
        }  else if (strcmp(input, "ffnn") == 0) {
            ffnntest();
        }
    }

    return 0;
}
