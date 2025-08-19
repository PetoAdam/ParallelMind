#ifndef PARALLELMIND_MATRIX_H
#define PARALLELMIND_MATRIX_H

#include <vector>
#include <cuda_runtime.h>

class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    ~Matrix();

    // Rule of five
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;

    float* data() const;
    size_t rows() const;
    size_t cols() const;

    bool set(size_t row, size_t col, float value);
    float operator()(size_t row, size_t col) const;

    // Fills the matrix with random values
    void randomize();

    // Matrix multiplication (CUDA-accelerated)
    static Matrix multiply(const Matrix& A, const Matrix& B);

    // Bulk copy helpers
    void copyToHost(float* dst, size_t count) const;
    void copyFromHost(const float* src, size_t count);

private:
    size_t _rows;
    size_t _cols;
    float* _data; // Device pointer for GPU memory

    // Internal utility for allocating GPU memory
    void allocate();
};

// GPU helper ops (implemented in Matrix.cu)
void vecUpdate(float* w, const float* x, float scale, size_t n);
void vecAccumulate(float* out, const float* w, float scale, size_t n);
void addBias(float* y, const float* b, size_t n);
void reluInplace(float* y, size_t n);
void sigmoidInplace(float* y, size_t n);
void computeDelta(float* delta, const float* error, const float* activated, int mode, size_t n);

#endif // PARALLELMIND_MATRIX_H
