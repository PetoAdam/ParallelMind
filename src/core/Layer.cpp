#include "Layer.h"
#include <stdexcept>
#include <iostream>

Layer::Layer(size_t numNodes, size_t inputSize)
    : _numNodes(numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        _nodes.push_back(Node(inputSize));  // Create nodes with the specified input size
    }
}

Layer::~Layer() {}

Matrix Layer::forward(const Matrix& input) {
    // Ensure the input size matches the layer's expected input size
    if (input.rows() != _nodes[0].getInputSize() || input.cols() != 1) {
        throw std::invalid_argument("Input size does not match the expected size of the layer.");
    }

    std::cout << "Input size is correct. Proceeding with forward pass..." << std::endl;

    // Pass the input through each node in the layer
    Matrix output(_numNodes, 1);  // Output matrix has the size of the number of nodes
    for (size_t i = 0; i < _numNodes; ++i) {
        std::cout << "Processing node " << i << std::endl;
        _nodes[i].setInput(input);  // Set input for each node
        // TODO: batch data, set uses cudaMemCopy, which is more effective with larger chunks of data at once
        // This is always 1 column, so we don't have to worry about not continuous memory
        output.set(i, 0, _nodes[i].activate()); // Activate node and store result
    }

    std::cout << "Forward pass completed." << std::endl;
    return output;
}

Matrix Layer::backward(const Matrix& error, float learningRate) {
    // Calculate gradients for each node and propagate the error back through the layer
    Matrix gradient(_numNodes, 1);  // Gradient matrix

    // Calculate gradient for each node in the layer
    for (size_t i = 0; i < _numNodes; ++i) {
        gradient.set(i, 0, _nodes[i].computeGradient(error(i, 0)));
    }

    // Update the weights for each node
    for (size_t i = 0; i < _numNodes; ++i) {
        _nodes[i].updateWeights(learningRate);
    }

    return gradient;  // Return error propagated back to the previous layer
}


size_t Layer::getNumNodes() const {
    return _numNodes;
}
