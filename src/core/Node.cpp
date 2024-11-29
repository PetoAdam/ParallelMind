#include "Node.h"
#include <cmath>
#include <random>
#include <ctime>
#include <stdexcept>
#include <iostream>

Node::Node(size_t inputSize)
    : _inputSize(inputSize), _weights(1, inputSize), _input(inputSize, 1), _bias(0.0f) {
    _weights.randomize();
    std::srand(std::time(nullptr));
    _bias = static_cast<float>(std::rand()) / RAND_MAX;
}

Node::~Node() {}

void Node::setInput(const Matrix& input) {
    if (input.rows() != _inputSize || input.cols() != 1) {
        throw std::invalid_argument("Input dimensions do not match the Node's expected size.");
    }
    std::cout << "Setting input..." << std::endl;
    cudaMemcpy(_input.data(), input.data(), _inputSize * sizeof(float), cudaMemcpyDeviceToDevice);
    std::cout << "Input set successfully" << std::endl;
}

float Node::activate() const {
    Matrix result = Matrix::multiply(_weights, _input);

    float hostResult;
    cudaMemcpy(&hostResult, result.data(), sizeof(float), cudaMemcpyDeviceToHost);

    return sigmoid(hostResult + _bias);
}

float Node::sigmoid(float x) const {
    return 1.0f / (1.0f + std::exp(-x));
}

float Node::computeGradient(float error) {
    // Compute gradient using chain rule, etc. (this is a simple example)
    return error * activate() * (1 - activate());
}

void Node::updateWeights(float learningRate) {
    // Use gradient descent to update the weights (simplified)
    for (size_t i = 0; i < _inputSize; ++i) {
        _weights.set(i, 0, _weights(i, 0) - learningRate * _input(i, 0));
    }
}
