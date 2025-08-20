#include "Matrix.h"
#include <cstdlib>
#include <ctime>
#include <random>
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
    if (_data) cudaFree(_data);
}

Matrix::Matrix(const Matrix& other) : _rows(other._rows), _cols(other._cols) {
    allocate();
    cudaMemcpy(_data, other._data, _rows*_cols*sizeof(float), cudaMemcpyDeviceToDevice);
}

Matrix::Matrix(Matrix&& other) noexcept : _rows(other._rows), _cols(other._cols), _data(other._data) {
    other._data = nullptr;
    other._rows = 0; other._cols = 0;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;
    if (_data) cudaFree(_data);
    _rows = other._rows; _cols = other._cols;
    allocate();
    cudaMemcpy(_data, other._data, _rows*_cols*sizeof(float), cudaMemcpyDeviceToDevice);
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this == &other) return *this;
    if (_data) cudaFree(_data);
    _rows = other._rows; _cols = other._cols;
    _data = other._data;
    other._data = nullptr; other._rows = 0; other._cols = 0;
    return *this;
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
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<float> hostData(_rows * _cols);
    for (auto& value : hostData) value = dist(gen);
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
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Matrix multiply kernel failed");
    }

    return C;
}

void Matrix::copyToHost(float* dst, size_t count) const {
    cudaMemcpy(dst, _data, count * sizeof(float), cudaMemcpyDeviceToHost);
}

void Matrix::copyFromHost(const float* src, size_t count) {
    cudaMemcpy(_data, src, count * sizeof(float), cudaMemcpyHostToDevice);
}

// Vector ops kernels
__global__ void vecUpdateKernel(float* w, const float* x, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        w[idx] += scale * x[idx];
    }
}

__global__ void vecAccumulateKernel(float* out, const float* w, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] += scale * w[idx];
    }
}

void vecUpdate(float* w, const float* x, float scale, size_t n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    vecUpdateKernel<<<grid, block>>>(w, x, scale, n);
    cudaDeviceSynchronize();
}

void vecAccumulate(float* out, const float* w, float scale, size_t n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    vecAccumulateKernel<<<grid, block>>>(out, w, scale, n);
    cudaDeviceSynchronize();
}

// Bias add and activation kernels
__global__ void addBiasKernel(float* y, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] += b[idx];
}

__global__ void reluInplaceKernel(float* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = y[idx] > 0.0f ? y[idx] : 0.0f;
}

__device__ inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void sigmoidInplaceKernel(float* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = sigmoidf(y[idx]);
}

__global__ void deltaKernel(float* delta, const float* error, const float* activated, int mode, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float deriv = 1.0f;
        if (mode == 0) { // ReLU
            deriv = activated[idx] > 0.0f ? 1.0f : 0.0f;
        } else if (mode == 1) { // Sigmoid
            float a = activated[idx];
            deriv = a * (1.0f - a);
        } else { // Linear
            deriv = 1.0f;
        }
        delta[idx] = error[idx] * deriv;
    }
}

void addBias(float* y, const float* b, size_t n) {
    dim3 block(256); dim3 grid((n + block.x - 1)/block.x);
    addBiasKernel<<<grid, block>>>(y, b, n);
    cudaDeviceSynchronize();
}

void reluInplace(float* y, size_t n) {
    dim3 block(256); dim3 grid((n + block.x - 1)/block.x);
    reluInplaceKernel<<<grid, block>>>(y, n);
    cudaDeviceSynchronize();
}

void sigmoidInplace(float* y, size_t n) {
    dim3 block(256); dim3 grid((n + block.x - 1)/block.x);
    sigmoidInplaceKernel<<<grid, block>>>(y, n);
    cudaDeviceSynchronize();
}

void computeDelta(float* delta, const float* error, const float* activated, int mode, size_t n) {
    dim3 block(256); dim3 grid((n + block.x - 1)/block.x);
    deltaKernel<<<grid, block>>>(delta, error, activated, mode, n);
    cudaDeviceSynchronize();
}

// Softmax: compute numerically stable softmax for a single vector
__global__ void softmaxKernel(float* x, size_t n, float maxval, float sumexp) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = expf(x[idx] - maxval) / sumexp;
    }
}

void softmaxInplace(float* logits, size_t n) {
    // Copy to host to compute max and sumexp cheaply (n is output size, typically small)
    std::vector<float> h(n);
    cudaMemcpy(h.data(), logits, n*sizeof(float), cudaMemcpyDeviceToHost);
    float maxv = -1e30f; for (size_t i=0;i<n;++i) if (h[i] > maxv) maxv = h[i];
    float sumexp = 0.0f; for (size_t i=0;i<n;++i) sumexp += expf(h[i] - maxv);
    dim3 block(256); dim3 grid((n + block.x - 1)/block.x);
    softmaxKernel<<<grid, block>>>(logits, n, maxv, sumexp);
    cudaDeviceSynchronize();
}

// grad = softmax(logits) - target
void softmaxCrossEntropyGrad(float* grad, const float* logits, const float* target, size_t n) {
    // Compute softmax(logits) into grad buffer, then subtract target
    cudaMemcpy(grad, logits, n*sizeof(float), cudaMemcpyDeviceToDevice);
    softmaxInplace(grad, n);
    // grad -= target
    dim3 block(256); dim3 grid((n + block.x - 1)/block.x);
    vecAccumulateKernel<<<grid, block>>>(grad, target, -1.0f, n);
    cudaDeviceSynchronize();
}
