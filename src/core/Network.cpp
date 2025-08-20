#include "Network.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <fstream>
#include <cstring>
#include <sstream>

Network::Network(const std::vector<size_t>& layerSizes) : _numLayers(layerSizes.size()) {
    for (size_t i = 0; i < _numLayers - 1; ++i) {
        ActivationType act = (i < _numLayers - 2) ? ActivationType::ReLU : ActivationType::Linear;
        _layers.emplace_back(Layer(layerSizes[i + 1], layerSizes[i], act));
    }
}

Network::~Network() {}
std::string Network::summary() const {
    std::ostringstream oss;
    oss << "Network[";
    if (_layers.empty()) { oss << "]"; return oss.str(); }
    // Input size is _layers[0].getInputSize()
    oss << _layers[0].getInputSize();
    for (size_t i=0;i<_layers.size();++i) {
        oss << "-" << _layers[i].getNumNodes();
    }
    oss << "] activations=";
    for (size_t i=0;i<_layers.size();++i) {
        auto a = _layers[i].activation();
        const char* an = (a==ActivationType::ReLU?"ReLU":(a==ActivationType::Sigmoid?"Sigmoid":(a==ActivationType::Linear?"Linear":"Softmax")));
        oss << (i?",":"") << an;
    }
    return oss.str();
}

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

    if (target.rows() != output.rows() || target.cols() != 1) {
        throw std::invalid_argument("Target shape mismatch");
    }
    Matrix outputError(output.rows(), 1);
    // If last layer is Softmax, compute grad = softmax(logits) - target;
    // Otherwise, use simple diff (output - target)
    if (!_layers.empty() && _layers.back().activation() == ActivationType::Softmax) {
        // output currently holds activation (since Layer::forward applies softmax if set)
        // We need logits to compute stable grad; approximate using activated values: grad = activated - target
        for (size_t i = 0; i < output.rows(); ++i) {
            outputError.set(i, 0, output(i, 0) - target(i, 0));
        }
    } else {
        for (size_t i = 0; i < output.rows(); ++i) {
            outputError.set(i, 0, output(i, 0) - target(i, 0));
        }
    }

    // Backpropagate error through layers (starting from the output layer)
    Matrix currentError = outputError;
    for (size_t idx = _layers.size(); idx-- > 0;) {
        Layer& layer = _layers[idx];
        Matrix previousError = layer.backward(currentError, learningRate);
        currentError = previousError;
    }
}

void Network::train(const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets, size_t epochs, float learningRate) {
    std::cout << "[Train] " << summary() << ", optimizer=SGD(batch=1), epochs=" << epochs << ", lr=" << learningRate << "\n";
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle indices for SGD
        std::vector<size_t> idx(inputs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(42 + epoch);
        std::shuffle(idx.begin(), idx.end(), rng);

        size_t correct = 0; double cumLoss = 0.0;
        for (size_t k = 0; k < idx.size(); ++k) {
            size_t i = idx[k];
            backpropagate(inputs[i], targets[i], learningRate);
            Matrix out = forward(inputs[i]);
            size_t pred = 0, gt = 0; float best = -1e9f;
            for (size_t r = 0; r < out.rows(); ++r) {
                float v = out(r,0);
                if (v > best) { best = v; pred = r; }
                if (targets[i](r,0) > 0.5f) gt = r;
            }
            if (pred == gt) correct++;
            // Accumulate simple loss (MSE as a proxy)
            float loss = 0.0f;
            for (size_t r = 0; r < out.rows(); ++r) {
                float diff = out(r,0) - targets[i](r,0);
                loss += diff * diff;
            }
            cumLoss += loss;
        }
        double acc = inputs.empty()?0.0:(double)correct/inputs.size();
        double avgLoss = inputs.empty()?0.0:cumLoss/inputs.size();
        std::cout << "[Epoch " << (epoch+1) << "] acc=" << acc << ", loss=" << avgLoss << "\n";
    }
}

void Network::trainMiniBatch(const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets,
                        size_t epochs, float learningRate, size_t batchSize) {
    if (inputs.size() != targets.size()) throw std::invalid_argument("inputs/targets size mismatch");
    const size_t N = inputs.size();
    std::cout << "[Train] " << summary() << ", optimizer=SGD(batch=" << (batchSize?batchSize:1) << "), epochs=" << epochs << ", lr=" << learningRate << "\n";
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle indices
        std::vector<size_t> idx(N); std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(42 + epoch); std::shuffle(idx.begin(), idx.end(), rng);
        size_t correct = 0; double cumLoss = 0.0;
        for (size_t start = 0; start < N; start += batchSize) {
            size_t end = std::min(N, start + batchSize);
            // Simple mini-batch: accumulate weight updates over the batch (SGD-style average)
            for (size_t k = start; k < end; ++k) {
                size_t i = idx[k];
                backpropagate(inputs[i], targets[i], learningRate / static_cast<float>(end - start));
                Matrix out = forward(inputs[i]);
                size_t pred = 0, gt = 0; float best = -1e9f;
                for (size_t r = 0; r < out.rows(); ++r) {
                    float v = out(r,0);
                    if (v > best) { best = v; pred = r; }
                    if (targets[i](r,0) > 0.5f) gt = r;
                }
                if (pred == gt) correct++;
                // accumulate loss (MSE proxy)
                float loss = 0.0f;
                for (size_t r = 0; r < out.rows(); ++r) {
                    float diff = out(r,0) - targets[i](r,0);
                    loss += diff * diff;
                }
                cumLoss += loss;
            }
        }
        double acc = N? (double)correct/N : 0.0;
        double avgLoss = N? cumLoss/N : 0.0;
        std::cout << "[Epoch " << (epoch+1) << "] acc=" << acc << ", loss=" << avgLoss << "\n";
    }
}

void Network::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("failed to open model for write");
    const char magic[8] = {'P','M','N','N','M','O','D','L'};
    f.write(magic, 8);
    uint64_t L = _layers.size();
    f.write(reinterpret_cast<const char*>(&L), sizeof(L));
    for (const auto& layer : _layers) {
        uint64_t M = layer.getNumNodes();
        uint64_t N = layer.getInputSize();
        int act = static_cast<int>(layer.activation());
        f.write(reinterpret_cast<const char*>(&M), sizeof(M));
        f.write(reinterpret_cast<const char*>(&N), sizeof(N));
        f.write(reinterpret_cast<const char*>(&act), sizeof(act));
        std::vector<float> W, b;
        layer.getWeightsHost(W);
        layer.getBiasHost(b);
        f.write(reinterpret_cast<const char*>(W.data()), W.size()*sizeof(float));
        f.write(reinterpret_cast<const char*>(b.data()), b.size()*sizeof(float));
    }
}

Network Network::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("failed to open model for read");
    char magic[8]; f.read(magic, 8);
    const char expect[8] = {'P','M','N','N','M','O','D','L'};
    if (std::memcmp(magic, expect, 8) != 0) throw std::runtime_error("bad model file");
    uint64_t L; f.read(reinterpret_cast<char*>(&L), sizeof(L));
    std::vector<size_t> sizes; sizes.reserve(L+1);
    std::vector<ActivationType> acts; acts.reserve(L);
    struct LayerBuf { uint64_t M, N; int act; std::vector<float> W, b; };
    std::vector<LayerBuf> bufs; bufs.reserve(L);
    for (uint64_t i=0;i<L;++i) {
        LayerBuf lb; f.read(reinterpret_cast<char*>(&lb.M), sizeof(lb.M)); f.read(reinterpret_cast<char*>(&lb.N), sizeof(lb.N)); f.read(reinterpret_cast<char*>(&lb.act), sizeof(lb.act));
        lb.W.resize(lb.M*lb.N); lb.b.resize(lb.M);
        f.read(reinterpret_cast<char*>(lb.W.data()), lb.W.size()*sizeof(float));
        f.read(reinterpret_cast<char*>(lb.b.data()), lb.b.size()*sizeof(float));
        bufs.emplace_back(std::move(lb));
    }
    sizes.push_back(bufs[0].N);
    for (auto& lb : bufs) sizes.push_back(lb.M), acts.push_back(static_cast<ActivationType>(lb.act));
    Network net(sizes);
    for (size_t i=0;i<bufs.size();++i) {
        // Ensure activation matches constructed
        // Overwrite weights/bias
        net._layers[i].setWeightsHost(bufs[i].W);
        net._layers[i].setBiasHost(bufs[i].b);
    }
    return net;
}
