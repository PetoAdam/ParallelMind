#include "Network.h"
#include <iostream>
#include <stdexcept>

Network::Network(const std::vector<size_t>& layerSizes) : _numLayers(layerSizes.size()) {
    for (size_t i = 0; i < _numLayers - 1; ++i) {
        _layers.push_back(Layer(layerSizes[i + 1], layerSizes[i]));  // Layer sizes are input/output
    }
}

Network::~Network() {}

Matrix Network::forward(const Matrix& input) {
    Matrix currentInput = input;
    for (size_t i = 0; i < _numLayers - 1; ++i) {
        currentInput = _layers[i].forward(currentInput);  // Forward pass through each layer
    }
    return currentInput;  // Return final output from the last layer
}

void Network::backpropagate(const Matrix& input, const Matrix& target, float learningRate) {
    // Perform forward pass first
    Matrix output = forward(input);

    // Compute output layer error (target - output)
    Matrix outputError(output.rows(), output.cols());
    for (size_t i = 0; i < output.rows(); ++i) {
        outputError.set(i, 0, target(i, 0) - output(i, 0));
    }

    // Backpropagate error through layers (starting from the output layer)
    Matrix currentError = outputError;
    for (size_t i = _numLayers - 2; i >= 0; --i) {
        Layer& layer = _layers[i];

        // Compute error for the previous layer (using current layer's error)
        Matrix previousError = layer.backward(currentError, learningRate);

        currentError = previousError;  // Set current error for next iteration
    }
}

void Network::train(const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets, size_t epochs, float learningRate) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Train on each input-target pair
            backpropagate(inputs[i], targets[i], learningRate);
        }
        std::cout << "Epoch " << epoch + 1 << " complete" << std::endl;
    }
}
