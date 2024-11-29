#include <iostream>
#include <cuda_runtime.h>

// Kernel function to print "Hello World" from the GPU
__global__ void helloWorldKernel() {
    printf("Hello World from GPU! Thread: [%d, %d]\n", threadIdx.x, blockIdx.x);
}

int main() {
    // Launch the kernel with 1 block of 10 threads
    helloWorldKernel<<<1, 10>>>();

    // Wait for the GPU to finish before accessing on the host
    cudaDeviceSynchronize();

    // Check for any errors during execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Hello World from CPU!" << std::endl;
    return 0;
}
