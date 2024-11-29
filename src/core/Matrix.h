#ifndef PARALLELMIND_MATRIX_H
#define PARALLELMIND_MATRIX_H

#include <vector>
#include <cuda_runtime.h>

class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    ~Matrix();

    float* data() const;
    size_t rows() const;
    size_t cols() const;

    float& operator()(size_t row, size_t col);
    float operator()(size_t row, size_t col) const;

    // Fills the matrix with random values
    void randomize();

    // Matrix multiplication (CUDA-accelerated)
    static Matrix multiply(const Matrix& A, const Matrix& B);

private:
    size_t _rows;
    size_t _cols;
    float* _data; // Device pointer for GPU memory

    // Internal utility for allocating GPU memory
    void allocate();
};

#endif // PARALLELMIND_MATRIX_H
