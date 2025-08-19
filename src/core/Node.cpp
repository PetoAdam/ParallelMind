#include "Node.h"
#include <cmath>
#include <random>
#include <ctime>
#include <stdexcept>
#include <iostream>

Node::Node(size_t inputSize, ActivationType activation)
    : _inputSize(inputSize), _weights(1, inputSize), _input(inputSize, 1), _bias(0.0f), _activationType(activation) {
    _weights.randomize();
    // Scale weights for He init
    float scale = std::sqrt(2.0f / static_cast<float>(inputSize));
    for (size_t i = 0; i < inputSize; ++i) {
        float w = _weights(0, i);
        w *= scale;
        _weights.set(0, i, w);
    }
    // Small positive bias helps ReLU
    _bias = (_activationType == ActivationType::ReLU) ? 0.01f : 0.0f;
}

Node::~Node() {}

void Node::setInput(const Matrix& input) {
    if (input.rows() != _inputSize || input.cols() != 1) {
        throw std::invalid_argument("Input dimensions do not match the Node's expected size.");
    }
    cudaMemcpy(_input.data(), input.data(), _inputSize * sizeof(float), cudaMemcpyDeviceToDevice);
}

float Node::activate() {
    Matrix result = Matrix::multiply(_weights, _input);

    float hostResult;
    cudaMemcpy(&hostResult, result.data(), sizeof(float), cudaMemcpyDeviceToHost);
    _lastZ = hostResult + _bias;
    switch (_activationType) {
        case ActivationType::Sigmoid: _lastActivation = sigmoid(_lastZ); break;
        case ActivationType::ReLU: _lastActivation = relu(_lastZ); break;
        case ActivationType::Linear: _lastActivation = _lastZ; break;
    }
    return _lastActivation;
}

float Node::sigmoid(float x) const {
    return 1.0f / (1.0f + std::exp(-x));
}

float Node::computeGradient(float error) {
    float deriv;
    switch (_activationType) {
        case ActivationType::Sigmoid: deriv = _lastActivation * (1.0f - _lastActivation); break;
        case ActivationType::ReLU: deriv = _lastZ > 0.0f ? 1.0f : 0.0f; break;
        case ActivationType::Linear: deriv = 1.0f; break;
    }
    _lastDelta = error * deriv; // dL/dz
    return _lastDelta;
}

void Node::updateWeights(float learningRate) {
    // w := w - lr * (delta * x^T), b := b - lr * delta
    // Vectorized on GPU: w += scale * x, where scale = -lr * delta
    float scale = -learningRate * _lastDelta;
    vecUpdate(_weights.data(), _input.data(), scale, _inputSize);
    _bias -= learningRate * _lastDelta;
}

float Node::getWeight(size_t index) const {
    if (index >= _inputSize) throw std::out_of_range("weight index out of range");
    float w = _weights(0, index);
    return w;
}
