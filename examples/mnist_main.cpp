#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include "../src/core/Network.h"
#include "../src/utils/NpyReader.hpp"

static void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [--train] [--load <model.pmmdl>] [--save <model.pmmdl>]\n"
              << "            [--out-act softmax|relu|sigmoid|linear] [--hid-act relu|sigmoid|linear]\n"
              << "            [--batch N] [--epochs K] [--lr X.Y] [--help]\n\n"
              << "Flags:\n"
              << "  --train                 Train a new model (default if neither --train nor --load given)\n"
              << "  --load <path>           Load an existing model from <path>\n"
              << "  --save <path>           Save trained model to <path> (default: /workspace/build/mnist.pmmdl)\n"
              << "  --out-act <act>         Output activation: softmax|relu|sigmoid|linear (default: softmax)\n"
              << "  --hid-act <act>         Hidden activation: relu|sigmoid|linear (default: relu)\n"
              << "  --batch N               Mini-batch size (0 = per-sample SGD, default: 0)\n"
              << "  --epochs K              Number of epochs (default: 5)\n"
              << "  --lr X.Y                Learning rate (default: 0.01)\n"
              << "  --help                  Show this help and exit\n\n"
              << "Examples:\n"
              << "  " << prog << " --train --out-act softmax --hid-act relu --batch 128 --epochs 5 --lr 0.01\n"
              << "  " << prog << " --load /workspace/build/mnist.pmmdl\n";
}

static void copyHostToDevice(Matrix& m, const std::vector<float>& host) {
    cudaMemcpy(m.data(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
}

static void makeOneHot(Matrix& m, unsigned char label) {
    for (size_t r = 0; r < m.rows(); ++r) m.set(r, 0, 0.0f);
    m.set(static_cast<size_t>(label), 0, 1.0f);
}

static size_t argmaxDeviceVector(const Matrix& m) {
    size_t bestIdx = 0; float bestVal = -1e9f;
    for (size_t r = 0; r < m.rows(); ++r) {
        float v = m(r,0);
        if (v > bestVal) { bestVal = v; bestIdx = r; }
    }
    return bestIdx;
}

static void printAsciiDigit(const std::vector<float>& img, int H=28, int W=28, float thresh=0.5f) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) std::cout << (img[y*W+x] > thresh ? '#' : '.');
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    std::string mode = "train"; // train or load
    std::string modelPath = "/workspace/build/mnist.pmmdl";
    std::string outAct = "softmax"; // output activation
    std::string hidAct = "relu";    // hidden activation
    size_t batchSize = 0; // 0 = per-sample SGD
    size_t epochs = 5;    // default epochs
    float lr = 0.01f;     // default learning rate
    // quick pre-scan for help
    for (int i=1;i<argc;++i) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            printHelp(argv[0]);
            return 0;
        }
    }
    for (int i=1;i<argc;++i) {
        std::string arg = argv[i];
        if (arg == "--train") mode = "train";
        else if (arg == "--load" && i+1 < argc) { mode = "load"; modelPath = argv[++i]; }
        else if (arg == "--save" && i+1 < argc) { modelPath = argv[++i]; }
        else if (arg == "--out-act" && i+1 < argc) { outAct = argv[++i]; }
        else if (arg == "--hid-act" && i+1 < argc) { hidAct = argv[++i]; }
        else if (arg == "--batch" && i+1 < argc) { batchSize = static_cast<size_t>(std::stoul(argv[++i])); }
        else if (arg == "--epochs" && i+1 < argc) { epochs = static_cast<size_t>(std::stoul(argv[++i])); }
        else if (arg == "--lr" && i+1 < argc) { lr = std::stof(argv[++i]); }
    }
    Network net({784, 128, 10});
    auto toAct = [](const std::string& s){
        std::string t=s; std::transform(t.begin(), t.end(), t.begin(), ::tolower);
        if (t=="relu") return ActivationType::ReLU;
        if (t=="sigmoid") return ActivationType::Sigmoid;
        if (t=="linear") return ActivationType::Linear;
        if (t=="softmax") return ActivationType::Softmax;
        return ActivationType::Linear;
    };
    net.setHiddenActivation(toAct(hidAct));
    net.setOutputActivation(toAct(outAct));

    NpyReader* reader = NpyReader::getInstance();
    std::vector<size_t> tr_img_shape; auto tr_images = reader->loadFloat("/workspace/data/mnist/mnist_train_images.npy", tr_img_shape);
    std::vector<size_t> tr_lbl_shape; auto tr_labels = reader->loadUint8("/workspace/data/mnist/mnist_train_labels.npy", tr_lbl_shape);
    assert(tr_img_shape.size()==3 && tr_lbl_shape.size()==1);
    size_t tr_n = tr_img_shape[0]; size_t H = tr_img_shape[1]; size_t W = tr_img_shape[2]; size_t image_size = H*W;
    std::vector<size_t> te_img_shape; auto te_images = reader->loadFloat("/workspace/data/mnist/mnist_test_images.npy", te_img_shape);
    std::vector<size_t> te_lbl_shape; auto te_labels = reader->loadUint8("/workspace/data/mnist/mnist_test_labels.npy", te_lbl_shape);
    assert(te_img_shape.size()==3 && te_lbl_shape.size()==1);
    size_t te_n = te_img_shape[0];

    size_t train_count = std::min<size_t>(5000, tr_n);
    size_t test_count  = std::min<size_t>(1000, te_n);
    std::vector<Matrix> trainX, trainY;
    trainX.reserve(train_count); trainY.reserve(train_count);
    for (size_t i=0;i<train_count;++i) {
        Matrix x(784,1); Matrix y(10,1);
        std::vector<float> flat(image_size);
        std::copy(tr_images.begin()+i*image_size, tr_images.begin()+(i+1)*image_size, flat.begin());
        for (auto &v : flat) v = std::clamp(v, 0.0f, 1.0f);
        copyHostToDevice(x, flat);
        makeOneHot(y, tr_labels[i]);
        trainX.emplace_back(std::move(x));
        trainY.emplace_back(std::move(y));
    }

    if (mode == "train") {
        if (batchSize > 0) net.trainMiniBatch(trainX, trainY, epochs, lr, batchSize);
        else net.train(trainX, trainY, epochs, lr);
        try { net.save(modelPath); std::cout << "Saved model to: " << modelPath << "\n"; } catch (const std::exception& e) { std::cerr << "Save failed: " << e.what() << "\n"; }
    } else {
        try { net = Network::load(modelPath); std::cout << "Loaded model from: " << modelPath << "\n"; } catch (const std::exception& e) { std::cerr << "Load failed: " << e.what() << "\n"; return 1; }
    }

    size_t correct = 0;
    for (size_t i=0;i<test_count;++i) {
        Matrix x(784,1);
        std::vector<float> flat(image_size);
        std::copy(te_images.begin()+i*image_size, te_images.begin()+(i+1)*image_size, flat.begin());
        for (auto &v : flat) v = std::clamp(v, 0.0f, 1.0f);
        copyHostToDevice(x, flat);
        Matrix out = net.forward(x);
        size_t pred = argmaxDeviceVector(out);
        if (pred == te_labels[i]) correct++;
        std::cout << "Sample " << i << ": Pred=" << pred << ", True=" << (int)te_labels[i] << "\n";
        printAsciiDigit(flat);
        std::cout << "-----------------------------\n";
    }
    std::cout << "Test accuracy on " << test_count << " samples: " << (test_count? (double)correct/test_count:0.0) << "\n";
    return 0;
}
