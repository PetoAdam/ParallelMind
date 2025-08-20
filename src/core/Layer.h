#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"
#include "Activation.h"
#include <vector>

class Layer {
public:
    // Constructor: takes the number of nodes and the size of the input for the layer
    Layer(size_t numNodes, size_t inputSize, ActivationType activation = ActivationType::ReLU);

    // Destructor
    ~Layer();

    // Forward propagation
    Matrix forward(const Matrix& input);

    // Backpropagation
    // error: dL/dA for this layer [numNodes x 1]; returns dL/dInput [inputSize x 1]
    Matrix backward(const Matrix& error, float learningRate);

    // Number of nodes
    size_t getNumNodes() const;

    // Expected input size
    size_t getInputSize() const { return _inputSize; }

    // Activation type
    ActivationType activation() const { return _activation; }
    void setActivation(ActivationType a) { _activation = a; }

    // Serialization helpers
    void getWeightsHost(std::vector<float>& out) const;          // size: numNodes*inputSize
    void getBiasHost(std::vector<float>& out) const;              // size: numNodes
    void setWeightsHost(const std::vector<float>& in);           // size must match
    void setBiasHost(const std::vector<float>& in);              // size must match

private:
    size_t _numNodes;   // Number of nodes in the layer
    size_t _inputSize{0};
    ActivationType _activation{ActivationType::ReLU};
    // Parameters
    Matrix _W; // [numNodes x inputSize]
    Matrix _b; // [numNodes x 1]
    // Caches
    Matrix _lastInput; // [inputSize x 1]
    Matrix _lastZ;     // [numNodes x 1]
};

#endif
