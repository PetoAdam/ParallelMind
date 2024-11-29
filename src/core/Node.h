#ifndef PARALLELMIND_NODE_H
#define PARALLELMIND_NODE_H

#include "Matrix.h"
#include <vector>

class Node {
public:
    Node(size_t inputSize);
    ~Node();

    size_t getInputSize() const { return _inputSize; }

    void setInput(const Matrix& input);
    float activate() const;
    float computeGradient(float error);
    void updateWeights(float learningRate);

private:
    size_t _inputSize;
    Matrix _weights;
    Matrix _input;
    float _bias;

    float sigmoid(float x) const;
};

#endif // PARALLELMIND_NODE_H
