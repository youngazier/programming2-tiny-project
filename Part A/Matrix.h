#ifndef MATRIX_H
#define MATRIX_H

#include "Vector.h"

class Matrix {
private:
    int mNumRows, mNumCols;
    double** mData;

public:
    Matrix(int rows, int cols);
    Matrix(const Matrix& other);
    ~Matrix();

    Matrix& operator=(const Matrix& other);
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Vector operator*(const Vector& vec) const;
    Matrix operator*(double scalar) const;

    double& operator()(int i, int j);       // 1-based 
    double operator()(int i, int j) const;
    int numRows() const;
    int numCols() const;

    double determinant() const;
    Matrix inverse() const;
    Matrix pseudoInverse() const;

    void print() const;
};

#endif
