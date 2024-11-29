#include "../src/core/Matrix.h"
#include "../src/core/Node.h"
#include "../src/core/Layer.h"
#include <cassert>
#include <iostream>

void testLayerForward() {
    // Define a layer with 3 nodes, each with 2 inputs
    Layer layer(3, 2);
    
    // Define an input matrix with 2 elements (matching the input size)
    Matrix input(2, 1);
    float hostInput[] = {1.0, 2.0};
    cudaMemcpy(input.data(), hostInput, 2 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform forward pass
    Matrix output = layer.forward(input);
    
    // Ensure output size matches the number of nodes
    assert(output.rows() == 3 && output.cols() == 1);

    // Print out the output
    float hostOutput[3];
    cudaMemcpy(hostOutput, output.data(), 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Layer output: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << hostOutput[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Layer forward test passed!" << std::endl;
}

int main() {
    testLayerForward();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
