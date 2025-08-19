#ifndef NETWORK_H
#define NETWORK_H

#include "Layer.h"
#include <vector>
#include <string>

class Network {
public:
    // Constructor: takes a vector of layer sizes, including input, hidden, and output layers
    Network(const std::vector<size_t>& layerSizes);

    // Destructor
    ~Network();

    // Perform forward propagation
    Matrix forward(const Matrix& input);

    // Perform backpropagation and update weights
    void backpropagate(const Matrix& input, const Matrix& target, float learningRate);

    // Train the network on the MNIST dataset
    void train(const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets, size_t epochs, float learningRate);

    // Save / Load model (simple binary format)
    void save(const std::string& path) const;
    static Network load(const std::string& path);

private:
    std::vector<Layer> _layers;
    size_t _numLayers;
};

#endif // NETWORK_H
