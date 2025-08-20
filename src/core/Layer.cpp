#include "Layer.h"
#include <stdexcept>

Layer::Layer(size_t numNodes, size_t inputSize, ActivationType activation)
        : _numNodes(numNodes), _inputSize(inputSize), _activation(activation),
                    _W(numNodes, inputSize), _b(numNodes, 1), _lastInput(inputSize,1), _lastZ(numNodes,1) {
                _W.randomize();
                // zero bias
                std::vector<float> zeros(numNodes, 0.0f);
                _b.copyFromHost(zeros.data(), numNodes);
}

Layer::~Layer() {}

Matrix Layer::forward(const Matrix& input) {
    if (input.rows() != _inputSize || input.cols() != 1) {
        throw std::invalid_argument("Input size does not match the expected size of the layer.");
    }
    // Cache input
    cudaMemcpy(_lastInput.data(), input.data(), _inputSize*sizeof(float), cudaMemcpyDeviceToDevice);
    // z = W * x
    _lastZ = Matrix::multiply(_W, input);
    // y = z + b
    addBias(_lastZ.data(), _b.data(), _numNodes);
    // activation
    Matrix out = Matrix(_numNodes,1);
    cudaMemcpy(out.data(), _lastZ.data(), _numNodes*sizeof(float), cudaMemcpyDeviceToDevice);
    if (_activation == ActivationType::ReLU) {
        reluInplace(out.data(), _numNodes);
    } else if (_activation == ActivationType::Sigmoid) {
        sigmoidInplace(out.data(), _numNodes);
    } else if (_activation == ActivationType::Softmax) {
        softmaxInplace(out.data(), _numNodes);
    } // Linear: no-op
    return out;
}

Matrix Layer::backward(const Matrix& error, float learningRate) {
    // Compute delta = error * activation'(z)
    // For Softmax, when paired with cross-entropy, delta is provided by caller (Network)
    Matrix activated = forward(_lastInput); // activation of cached z
    Matrix delta(_numNodes,1);
    if (_activation == ActivationType::Softmax) {
        // delta = error directly
        cudaMemcpy(delta.data(), error.data(), _numNodes*sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        // mode: 0 ReLU, 1 Sigmoid, 2 Linear
        int mode = (_activation==ActivationType::ReLU?0:(_activation==ActivationType::Sigmoid?1:2));
        computeDelta(delta.data(), error.data(), activated.data(), mode, _numNodes);
    }

    // prevError = W^T * delta
    // W: [M x N], delta: [M x 1] -> prevError: [N x 1]
    // Instead of custom kernel, use multiply on transposed by constructing a temporary transposed Matrix (simple for 1D col vecs we can do accumulate kernel). For simplicity, use accumulate kernel per row.
    Matrix prevError(_inputSize,1);
    std::vector<float> zeros(_inputSize, 0.0f);
    prevError.copyFromHost(zeros.data(), _inputSize);
    // Accumulate prevError += delta[i] * W[i,:]^T
    // Copy delta to host for scale values
    std::vector<float> delta_h(_numNodes);
    delta.copyToHost(delta_h.data(), _numNodes);
    for (size_t i=0;i<_numNodes;++i) {
        vecAccumulate(prevError.data(), _W.data() + i*_inputSize, delta_h[i], _inputSize);
    }

    // Update weights and bias: W := W - lr * (delta * x^T), b := b - lr * delta
    // For each row i: W[i,:] += (-lr*delta[i]) * lastInput^T
    for (size_t i=0;i<_numNodes;++i) {
        vecUpdate(_W.data() + i*_inputSize, _lastInput.data(), -learningRate * delta_h[i], _inputSize);
    }
    // b update
    // Copy delta to host once (already have delta_h)
    std::vector<float> b_h(_numNodes);
    _b.copyToHost(b_h.data(), _numNodes);
    for (size_t i=0;i<_numNodes;++i) b_h[i] -= learningRate * delta_h[i];
    _b.copyFromHost(b_h.data(), _numNodes);

    return prevError;
}


size_t Layer::getNumNodes() const {
    return _numNodes;
}

void Layer::getWeightsHost(std::vector<float>& out) const {
    out.resize(_numNodes * _inputSize);
    _W.copyToHost(out.data(), out.size());
}

void Layer::getBiasHost(std::vector<float>& out) const {
    out.resize(_numNodes);
    _b.copyToHost(out.data(), out.size());
}

void Layer::setWeightsHost(const std::vector<float>& in) {
    if (in.size() != _numNodes * _inputSize) throw std::runtime_error("weights size mismatch");
    _W.copyFromHost(in.data(), in.size());
}

void Layer::setBiasHost(const std::vector<float>& in) {
    if (in.size() != _numNodes) throw std::runtime_error("bias size mismatch");
    _b.copyFromHost(in.data(), in.size());
}
