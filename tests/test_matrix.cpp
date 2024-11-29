#include "../src/core/Matrix.h"
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

int main() {
    testMatrixMultiplication();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
