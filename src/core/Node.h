#ifndef PARALLELMIND_NODE_H
#define PARALLELMIND_NODE_H

#include "Matrix.h"
#include "Activation.h"
#include <vector>

class Node {
public:
    Node(size_t inputSize, ActivationType activation = ActivationType::Sigmoid);
    ~Node();

    size_t getInputSize() const { return _inputSize; }

    void setInput(const Matrix& input);
    float activate();
    // error: dL/dA for this node's output; stores internal delta = error * sigmoid'(z)
    float computeGradient(float error);
    void updateWeights(float learningRate);

    // Access a single weight (host read)
    float getWeight(size_t index) const;

private:
    size_t _inputSize;
    Matrix _weights;
    Matrix _input;
    float _bias;
    float _lastActivation{0.0f};
    float _lastDelta{0.0f};
    float _lastZ{0.0f};
    ActivationType _activationType{ActivationType::Sigmoid};

    float sigmoid(float x) const;
    float relu(float x) const { return x > 0.0f ? x : 0.0f; }
};

#endif // PARALLELMIND_NODE_H
