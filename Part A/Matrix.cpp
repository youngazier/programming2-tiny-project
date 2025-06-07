#include "Matrix.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

Matrix::Matrix(int rows, int cols) : mNumRows(rows), mNumCols(cols) {
    mData = new double*[mNumRows];
    for (int i = 0; i < mNumRows; ++i) {
        mData[i] = new double[mNumCols];
        for (int j = 0; j < mNumCols; ++j)
            mData[i][j] = 0.0;
    }
}

Matrix::Matrix(const Matrix& other) : mNumRows(other.mNumRows), mNumCols(other.mNumCols) {
    mData = new double*[mNumRows];
    for (int i = 0; i < mNumRows; ++i) {
        mData[i] = new double[mNumCols];
        for (int j = 0; j < mNumCols; ++j)
            mData[i][j] = other.mData[i][j];
    }
}

Matrix::~Matrix() {
    for (int i = 0; i < mNumRows; ++i)
        delete[] mData[i];
    delete[] mData;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        for (int i = 0; i < mNumRows; ++i)
            delete[] mData[i];
        delete[] mData;

        mNumRows = other.mNumRows;
        mNumCols = other.mNumCols;
        mData = new double*[mNumRows];
        for (int i = 0; i < mNumRows; ++i) {
            mData[i] = new double[mNumCols];
            for (int j = 0; j < mNumCols; ++j)
                mData[i][j] = other.mData[i][j];
        }
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix& other) const {
    assert(mNumRows == other.mNumRows && mNumCols == other.mNumCols);
    Matrix result(mNumRows, mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] + other.mData[i][j];
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    assert(mNumRows == other.mNumRows && mNumCols == other.mNumCols);
    Matrix result(mNumRows, mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] - other.mData[i][j];
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    assert(mNumCols == other.mNumRows);
    Matrix result(mNumRows, other.mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < other.mNumCols; ++j)
            for (int k = 0; k < mNumCols; ++k)
                result.mData[i][j] += mData[i][k] * other.mData[k][j];
    return result;
}

Vector Matrix::operator*(const Vector& vec) const {
    assert(mNumCols == vec.size());
    Vector result(mNumRows);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result[i] += mData[i][j] * vec(j + 1);
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(mNumRows, mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] * scalar;
    return result;
}

double& Matrix::operator()(int i, int j) {
    return mData[i - 1][j - 1];
}

double Matrix::operator()(int i, int j) const {
    return mData[i - 1][j - 1];
}

int Matrix::numRows() const { return mNumRows; }
int Matrix::numCols() const { return mNumCols; }

void Matrix::print() const {
    for (int i = 0; i < mNumRows; ++i) {
        for (int j = 0; j < mNumCols; ++j)
            cout << mData[i][j] << " ";
        cout << endl;
    }
}

double Matrix::determinant() const {
    assert(mNumRows == mNumCols);
    if (mNumRows == 1) return mData[0][0];
    if (mNumRows == 2) return mData[0][0]*mData[1][1] - mData[0][1]*mData[1][0];

    double det = 0.0;
    for (int k = 0; k < mNumCols; ++k) {
        Matrix sub(mNumRows - 1, mNumCols - 1);
        for (int i = 1; i < mNumRows; ++i) {
            int sj = 0;
            for (int j = 0; j < mNumCols; ++j) {
                if (j == k) continue;
                sub.mData[i - 1][sj++] = mData[i][j];
            }
        }
        det += (k % 2 == 0 ? 1 : -1) * mData[0][k] * sub.determinant();
    }
    return det;
}

Matrix Matrix::inverse() const {
    assert(mNumRows == mNumCols);
    int n = mNumRows;
    Matrix A(*this);
    Matrix I(n, n);
    for (int i = 0; i < n; ++i) I.mData[i][i] = 1.0;

    for (int i = 0; i < n; ++i) {
        double pivot = A.mData[i][i];
        assert(fabs(pivot) > 1e-9);
        for (int j = 0; j < n; ++j) {
            A.mData[i][j] /= pivot;
            I.mData[i][j] /= pivot;
        }
        for (int k = 0; k < n; ++k) {
            if (k == i) continue;
            double factor = A.mData[k][i];
            for (int j = 0; j < n; ++j) {
                A.mData[k][j] -= factor * A.mData[i][j];
                I.mData[k][j] -= factor * I.mData[i][j];
            }
        }
    }
    return I;
}

Matrix Matrix::pseudoInverse() const {
    Matrix AT(mNumCols, mNumRows);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            AT.mData[j][i] = mData[i][j];

    Matrix ATA = AT * (*this);
    Matrix ATA_inv = ATA.inverse();
    return ATA_inv * AT;
}