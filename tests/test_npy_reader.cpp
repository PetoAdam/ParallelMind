#include "../src/utils/NpyReader.hpp"

void testNpyReader() {
    std::vector<int> expected_labels{5, 0, 4, 1, 9};
    try {
        // Get the singleton instance of NpyReader
        NpyReader* reader = NpyReader::getInstance();

        // Load image data (float32)
        std::vector<size_t> image_shape;
        auto images = reader->loadFloat("/workspace/data/mnist/mnist_train_images.npy", image_shape);
        assert(image_shape.size() == 3); // Expecting shape [N, H, W]
        size_t num_images = image_shape[0];
        size_t image_size = image_shape[1] * image_shape[2];

        // Validate and print the first few images
        for (size_t i = 0; i < std::min(num_images, size_t(5)); ++i) {
            std::cout << "Image " << i << ":\n";
            for (size_t y = 0; y < image_shape[1]; ++y) {
                for (size_t x = 0; x < image_shape[2]; ++x) {
                    std::cout << (images[i * image_size + y * image_shape[2] + x] > 0.5f ? "#" : ".");
                }
                std::cout << "\n";
            }
            std::cout << "----------\n";
        }

        // Load label data (uint8)
        std::vector<size_t> label_shape;
        auto labels = reader->loadUint8("/workspace/data/mnist/mnist_train_labels.npy", label_shape);
        assert(label_shape.size() == 1); // Expecting shape [N]

        // Validate and print the first few labels
        for (size_t i = 0; i < std::min(num_images, expected_labels.size()); ++i) {
            std::cout << "Label " << i << ": " << static_cast<int>(labels[i]) << std::endl;
            std::cout << "Expected label " << i << ": " << expected_labels[i] << std::endl;
            assert(static_cast<int>(labels[i]) == expected_labels[i]);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

int main() {
    testNpyReader();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}