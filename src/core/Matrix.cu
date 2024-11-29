#include "Matrix.h"
#include <cstdlib>
#include <ctime>
#include <iostream>

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, size_t rowsA, size_t colsA, size_t colsB) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float value = 0.0f;
        for (size_t k = 0; k < colsA; ++k) {
            value += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = value;
    }
}

Matrix::Matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols) {
    allocate();
}

Matrix::~Matrix() {
    cudaFree(_data);
}

float* Matrix::data() const {
    return _data;
}

size_t Matrix::rows() const {
    return _rows;
}

size_t Matrix::cols() const {
    return _cols;
}

bool Matrix::set(size_t row, size_t col, float value) {
    if(row >= _rows || col >= _cols) // Ensure within bounds
    {
        throw std::out_of_range("Matrix element index out of range");
    }
    cudaMemcpy(_data + row * _cols + col, &value, sizeof(float), cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize();
    return true;
}

float Matrix::operator()(size_t row, size_t col) const {
    if(row >= _rows || col >= _cols) // Ensure within bounds
    {
        throw std::out_of_range("Matrix element index out of range");
    }
    float value;
    cudaMemcpy(&value, _data + row * _cols + col, sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}

void Matrix::allocate() {
    cudaError_t err = cudaMalloc(&_data, _rows * _cols * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to allocate GPU memory.");
    }
}

void Matrix::randomize() {
    std::vector<float> hostData(_rows * _cols);
    std::srand(std::time(nullptr));
    for (auto& value : hostData) {
        value = static_cast<float>(std::rand()) / RAND_MAX;
    }
    cudaMemcpy(_data, hostData.data(), _rows * _cols * sizeof(float), cudaMemcpyHostToDevice);
}

Matrix Matrix::multiply(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    Matrix C(A.rows(), B.cols());

    dim3 blockSize(16, 16);
    dim3 gridSize((B.cols() + blockSize.x - 1) / blockSize.x,
                  (A.rows() + blockSize.y - 1) / blockSize.y);

    matrixMultiplyKernel<<<gridSize, blockSize>>>(A.data(), B.data(), C.data(), A.rows(), A.cols(), B.cols());
    cudaDeviceSynchronize();

    return C;
}
