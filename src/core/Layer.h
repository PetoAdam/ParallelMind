#ifndef LAYER_H
#define LAYER_H

#include "Node.h"
#include <vector>

class Layer {
public:
    // Constructor: takes the number of nodes and the size of the input for the layer
    Layer(size_t numNodes, size_t inputSize);

    // Destructor: cleanup
    ~Layer();

    // Perform forward propagation: pass input through the layer
    Matrix forward(const Matrix& input);

    // Perform back propagation
    Matrix backward(const Matrix& error, float learningRate);

    // Get the number of nodes in the layer
    size_t getNumNodes() const;

private:
    size_t _numNodes;   // Number of nodes in the layer
    std::vector<Node> _nodes;  // Vector of nodes in the layer
};

#endif
