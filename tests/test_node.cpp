#include "../src/core/Matrix.h"
#include "../src/core/Node.h"
#include <cassert>
#include <iostream>

void testNodeActivation() {
    // Define a Node with a single input
    Node node(3);
    
    // Define input matrix
    Matrix input(3, 1);
    float hostInput[] = {0.5, -1.5, 2.0};
    cudaMemcpy(input.data(), hostInput, 3 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set input and compute activation
    node.setInput(input);
    float output = node.activate();
    
    // Expected sigmoid output based on random weights and bias
    // NOTE: Since weights and bias are randomized, this test should focus on
    // verifying that the value falls within the expected range of the sigmoid function
    assert(output > 0.0f && output < 1.0f);

    std::cout << "Node activation output: " << output << std::endl;
    std::cout << "Node activation test passed!" << std::endl;
}

int main() {
    testNodeActivation();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
