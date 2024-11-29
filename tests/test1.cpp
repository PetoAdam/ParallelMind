#include "../src/core/Matrix.h"
#include "../src/core/Node.h"
#include "../src/core/Layer.h"
#include <cassert>
#include <iostream>

void testMatrixMultiplication() {
    // Define two small matrices for testing
    Matrix A(2, 3);
    Matrix B(3, 2);
    float hostA[] = {1, 2, 3, 4, 5, 6};
    float hostB[] = {7, 8, 9, 10, 11, 12};
    float expectedC[] = {58, 64, 139, 154}; // Precomputed: A * B

    // Copy host data to device matrices
    cudaMemcpy(A.data(), hostA, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data(), hostB, 6 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform multiplication
    Matrix C = Matrix::multiply(A, B);

    // Copy result back to host
    float hostC[4];
    cudaMemcpy(hostC, C.data(), 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < 4; ++i) {
        assert(abs(hostC[i] - expectedC[i]) < 1e-6);
    }

    std::cout << "Matrix multiplication test passed!" << std::endl;
}

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
    testMatrixMultiplication();
    testNodeActivation();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
