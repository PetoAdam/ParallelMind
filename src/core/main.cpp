#include <iostream>
#include <vector>
#include "Network.h"
#include "../utils/NpyReader.hpp"

int main()
{
    // Define the network architecture (e.g., 784 input nodes, 128 hidden nodes, 10 output nodes)
    Network net({784, 128, 10});

    // Assume we have preloaded MNIST dataset (inputs and targets) as matrices
    std::vector<Matrix> inputs;  // Input data (MNIST)
    std::vector<Matrix> targets; // Target labels (MNIST)

    // Load MNIST dataset into inputs and targets here
    // Get the singleton instance of NpyReader
    NpyReader *reader = NpyReader::getInstance();

    // Load image data (float32)
    std::vector<size_t> image_shape;
    auto images = reader->loadFloat("/workspace/data/mnist/mnist_train_images.npy", image_shape);
    assert(image_shape.size() == 3); // Expecting shape [N, H, W]
    size_t num_images = image_shape[0];
    size_t image_size = image_shape[1] * image_shape[2];

    // Load label data (uint8)
    std::vector<size_t> label_shape;
    auto labels = reader->loadUint8("/workspace/data/mnist/mnist_train_labels.npy", label_shape);
    assert(label_shape.size() == 1); // Expecting shape [N]

    // Transform images and labels into network-ready format
for (size_t i = 0; i < num_images; ++i) {
    // Flatten each image into a 784-element vector
    std::vector<float> flattened_image(image_size);
    for (size_t j = 0; j < image_size; ++j) {
        flattened_image[j] = images[i * image_size + j];
    }

    // Convert label into a one-hot encoded vector
    std::vector<float> one_hot_label(10, 0.0f);
    one_hot_label[labels[i]] = 1.0f;

    inputs.emplace_back(flattened_image);  // Add to inputs
    targets.emplace_back(one_hot_label);  // Add to targets
}

    // Train the network for 10 epochs with a learning rate of 0.01
    net.train(inputs, targets, 10, 0.01);

    return 0;
}
