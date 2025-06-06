#pragma once
#include <iostream>
#include <cmath>    // for fabs

#include "Vector.cpp"

class Matrix {
private:
    int     mNumRows;  // number of rows
    int     mNumCols;  // number of columns
    double **mData;    // pointer-to-pointer: mData[i] is row i

public:
    // Constructor: allocate mNumRows × mNumCols, initialize all entries to zero
    Matrix(int numRows, int numCols)
        : mNumRows(numRows), mNumCols(numCols), mData(nullptr)
    {
        if(numRows <= 0 || numCols <= 0) {
            std::cerr << "Matrix dimensions must be positive.\n";
            std::exit(EXIT_FAILURE);
        }
        mData = new double*[mNumRows];
        for(int i = 0; i < mNumRows; ++i) {
            mData[i] = new double[mNumCols];
        }
        // Initialize to zero
        for(int i = 0; i < mNumRows; ++i) {
            for(int j = 0; j < mNumCols; ++j) {
                mData[i][j] = 0.0;
            }
        }
    }

    // Copy constructor: deep copy
    Matrix(const Matrix &other)
        : mNumRows(other.mNumRows),
          mNumCols(other.mNumCols),
          mData(nullptr)
    {
        mData = new double*[mNumRows];
        for(int i = 0; i < mNumRows; ++i) {
            mData[i] = new double[mNumCols];
            for(int j = 0; j < mNumCols; ++j) {
                mData[i][j] = other.mData[i][j];
            }
        }
    }

    // Destructor: free memory
    ~Matrix()
    {
        for(int i = 0; i < mNumRows; ++i) {
            delete[] mData[i];
        }
        delete[] mData;
    }

    // Assignment operator: deep copy
    Matrix& operator=(const Matrix &other)
    {
        if(this == &other) return *this;
        if(mNumRows != other.mNumRows || mNumCols != other.mNumCols) {
            for(int i = 0; i < mNumRows; ++i) {
                delete[] mData[i];
            }
            delete[] mData;
            mNumRows = other.mNumRows;
            mNumCols = other.mNumCols;
            mData = new double*[mNumRows];
            for(int i = 0; i < mNumRows; ++i) {
                mData[i] = new double[mNumCols];
            }
        }
        for(int i = 0; i < mNumRows; ++i) {
            for(int j = 0; j < mNumCols; ++j) {
                mData[i][j] = other.mData[i][j];
            }
        }
        return *this;
    }

    // Accessors
    int numRows() const { return mNumRows; }
    int numCols() const { return mNumCols; }

    // One‐based indexing (no bound checks)
    double& operator()(int i, int j)
    {
        return mData[i - 1][j - 1];
    }
    const double& operator()(int i, int j) const
    {
        return mData[i - 1][j - 1];
    }

    // Matrix addition
    Matrix operator+(const Matrix &other) const
    {
        Matrix result(mNumRows, mNumCols);
        for(int i = 0; i < mNumRows; ++i) {
            for(int j = 0; j < mNumCols; ++j) {
                result.mData[i][j] = mData[i][j] + other.mData[i][j];
            }
        }
        return result;
    }

    // Matrix subtraction
    Matrix operator-(const Matrix &other) const
    {
        Matrix result(mNumRows, mNumCols);
        for(int i = 0; i < mNumRows; ++i) {
            for(int j = 0; j < mNumCols; ++j) {
                result.mData[i][j] = mData[i][j] - other.mData[i][j];
            }
        }
        return result;
    }

    // Matrix × Matrix multiplication
    Matrix operator*(const Matrix &other) const
    {
        // No dimension check; assumes mNumCols == other.mNumRows
        Matrix result(mNumRows, other.mNumCols);
        for(int i = 0; i < mNumRows; ++i) {
            for(int j = 0; j < other.mNumCols; ++j) {
                double sum = 0.0;
                for(int k = 0; k < mNumCols; ++k) {
                    sum += mData[i][k] * other.mData[k][j];
                }
                result.mData[i][j] = sum;
            }
        }
        return result;
    }

    // Scalar multiplication (matrix * scalar)
    Matrix operator*(double scalar) const
    {
        Matrix result(mNumRows, mNumCols);
        for(int i = 0; i < mNumRows; ++i) {
            for(int j = 0; j < mNumCols; ++j) {
                result.mData[i][j] = mData[i][j] * scalar;
            }
        }
        return result;
    }

    // Friend: scalar × matrix
    friend Matrix operator*(double scalar, const Matrix &mat)
    {
        return mat * scalar;
    }

    // Matrix × Vector multiplication
    Vector operator*(const Vector &vec) const
    {
        Vector result(mNumRows);
        for(int i = 0; i < mNumRows; ++i) {
            double sum = 0.0;
            for(int j = 0; j < mNumCols; ++j) {
                sum += mData[i][j] * vec[j];
            }
            result[i] = sum;
        }
        return result;
    }

    // Transpose
    Matrix transpose() const
    {
        Matrix T(mNumCols, mNumRows);
        for(int i = 0; i < mNumRows; ++i) {
            for(int j = 0; j < mNumCols; ++j) {
                T.mData[j][i] = mData[i][j];
            }
        }
        return T;
    }

    // Determinant using Gaussian elimination (square only)
    double determinant() const
    {
        if (mNumRows != mNumCols) {
            std::cerr << "Determinant requires a square matrix.\n";
            std::exit(EXIT_FAILURE);
        }
        int n = mNumRows;
        Matrix temp(*this);
        double det = 1.0;
        for(int i = 0; i < n; ++i) {
            int pivot = i;
            double maxAbs = std::fabs(temp.mData[i][i]);
            for(int row = i + 1; row < n; ++row) {
                if(std::fabs(temp.mData[row][i]) > maxAbs) {
                    maxAbs = std::fabs(temp.mData[row][i]);
                    pivot = row;
                }
            }
            if (std::fabs(temp.mData[pivot][i]) < 1e-12) {
                return 0.0;
            }
            if(pivot != i) {
                std::swap(temp.mData[i], temp.mData[pivot]);
                det = -det;
            }
            det *= temp.mData[i][i];
            for(int row = i + 1; row < n; ++row) {
                double factor = temp.mData[row][i] / temp.mData[i][i];
                for(int col = i; col < n; ++col) {
                    temp.mData[row][col] -= factor * temp.mData[i][col];
                }
            }
        }
        return det;
    }

    // Inverse using Gauss‐Jordan (square only)
    Matrix inverse() const
    {
        if (mNumRows != mNumCols) {
            std::cerr << "Inverse requires a square matrix.\n";
            std::exit(EXIT_FAILURE);
        }
        int n = mNumRows;
        Matrix aug(n, 2*n);
        for(int i = 0; i < n; ++i) {
            for(int j = 0; j < n; ++j) {
                aug.mData[i][j] = mData[i][j];
            }
            for(int j = n; j < 2*n; ++j) {
                aug.mData[i][j] = (j - n == i) ? 1.0 : 0.0;
            }
        }
        for(int i = 0; i < n; ++i) {
            int pivot = i;
            double maxAbs = std::fabs(aug.mData[i][i]);
            for(int row = i + 1; row < n; ++row) {
                if(std::fabs(aug.mData[row][i]) > maxAbs) {
                    maxAbs = std::fabs(aug.mData[row][i]);
                    pivot = row;
                }
            }
            if(std::fabs(aug.mData[pivot][i]) < 1e-12) {
                std::cerr << "Matrix is singular and cannot be inverted.\n";
                std::exit(EXIT_FAILURE);
            }
            if(pivot != i) {
                std::swap(aug.mData[i], aug.mData[pivot]);
            }
            double div = aug.mData[i][i];
            for(int col = 0; col < 2*n; ++col) {
                aug.mData[i][col] /= div;
            }
            for(int row = 0; row < n; ++row) {
                if(row != i) {
                    double factor = aug.mData[row][i];
                    for(int col = 0; col < 2*n; ++col) {
                        aug.mData[row][col] -= factor * aug.mData[i][col];
                    }
                }
            }
        }
        Matrix inv(n, n);
        for(int i = 0; i < n; ++i) {
            for(int j = 0; j < n; ++j) {
                inv.mData[i][j] = aug.mData[i][j + n];
            }
        }
        return inv;
    }

    // Moore‐Penrose pseudoinverse (full‐rank only)
    Matrix pseudoInverse() const
    {
        int m = mNumRows;
        int n = mNumCols;
        if(m >= n) {
            Matrix At = this->transpose();          // n×m
            Matrix AtA = At * (*this);               // n×n
            double det = AtA.determinant();
            if (std::fabs(det) < 1e-12) {
                std::cerr << "Cannot compute pseudoinverse: A^T A is singular.\n";
                std::exit(EXIT_FAILURE);
            }
            Matrix invAtA = AtA.inverse();          // n×n
            return invAtA * At;                     // n×m
        } else {
            Matrix At = this->transpose();          // n×m
            Matrix AAt = (*this) * At;               // m×m
            double det = AAt.determinant();
            if (std::fabs(det) < 1e-12) {
                std::cerr << "Cannot compute pseudoinverse: A A^T is singular.\n";
                std::exit(EXIT_FAILURE);
            }
            Matrix invAAt = AAt.inverse();          // m×m
            return At * invAAt;                     // n×m
        }
    }
};
