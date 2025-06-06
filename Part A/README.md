# Part A
## Overview
Part A focuses on implementing core linear algebra classes in C++ from scratch:
- `Vector`: Represents a methematical vector
- `Matrix`: Represents a 2D matrix with full support for algebraic operations
- `Linear System`: Solves linear systems using Gaussian elimination and least squares

These foundational classes are used in Part B to solve a real-world machine learning problem.

## `Vector.cpp`
### Constructor & Destructor
```cpp
Vector(int size)
        : mSize(size), mData(nullptr)
{
    if (size <= 0) {
        std::cerr << "Vector size must be positive.\n";
        std::exit(EXIT_FAILURE);
    }
    mData = new double[mSize];
    for(int i = 0; i < mSize; ++i) {
        mData[i] = 0.0;
    }
}

~Vector()
{
    delete[] mData;
}
```
Allocates a dynamic array and initializes all values to zero.

### Copy Constructor
Ensures **deep copy** when assigning or copying vectors
```cpp
Vector(const Vector &other)
    : mSize(other.mSize), mData(nullptr)
{
    mData = new double[mSize];
    for(int i = 0; i < mSize; ++i) {
        mData[i] = other.mData[i];
    }
}
```

### Indexing Operators
```cpp
double& operator[](int index) {
    return mData[index];
}

double& operator()(int i) {
    return mData[i - 1]; // One-based indexing
}
```
Provides both 0-based (`v[0]`) and 1-based (`v(1)`) access to elements, which is helpful for flexibility.

### Addition, Subtraction, and Scalar Multiplication
```cpp
Vector operator+(const Vector &other) const {
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i)
        result[i] = mData[i] + other[i];
    return result;
}

Vector operator-(const Vector &other) const {
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i)
        result[i] = mData[i] - other[i];
    return result;
}

Vector operator*(double scalar) const {
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i)
        result[i] = mData[i] * scalar;
    return result;
}
```

### Dot Product
```cpp
double dot(const Vector &other) const {
    double sum = 0.0;
    for (int i = 0; i < mSize; ++i)
        sum += mData[i] * other[i];
    return sum;
}
```
For example. computes the inner product:
$$\vec{a} \cdot \vec{b} = \sum_{i=1}^n a_i \cdot b_i$$

## `Matrix.cpp`
### Constructor & Destructor
```cpp
Matrix(int rows, int cols)
    : mNumRows(rows), mNumCols(cols)
{
    mData = new double*[rows];
    for (int i = 0; i < rows; ++i)
        mData[i] = new double[cols]();
}

~Matrix() {
    for (int i = 0; i < mNumRows; ++i)
        delete[] mData[i];
    delete[] mData;
}
```

### Transpose
```cpp
Matrix transpose() const {
    Matrix T(mNumCols, mNumRows);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            T.mData[j][i] = mData[i][j];
    return T;
}
```

### Matrix x Vector Multiplication
```cpp
Vector operator*(const Vector &vec) const {
    Vector result(mNumRows);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result[i] += mData[i][j] * vec[j];
    return result;
}
```
### Pseudoinverse
```cpp
Matrix pseudoInverse() const {
    Matrix At = this->transpose();
    Matrix AtA = At * (*this);
    Matrix invAtA = AtA.inverse();
    return invAtA * At;
}
```
Compute the **Moore-Penrose pseudoinverse**:
$$A^+ = (A^T A)^{-1}A^T$$

Used for solving overdetermined linear systems in least-squares regression.

## LinearSystem.cpp
### Solve Square Systems (Gaussian Elimination)
```cpp
Vector Solve() const {
    int n = mSize;
    Matrix aug(n, n + 1);
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j)
            aug(i, j) = (*mpA)(i, j);
        aug(i, n + 1) = (*mpb)(i);
    }
    for (int i = 1; i <= n; ++i) {
        for (int row = i + 1; row <= n; ++row) {
            double factor = aug(row, i) / aug(i, i);
            for (int col = i; col <= n + 1; ++col)
                aug(row, col) -= factor * aug(i, col);
        }
    }
    Vector x(n);
    for (int i = n; i >= 1; --i) {
        double sum = aug(i, n + 1);
        for (int j = i + 1; j <= n; ++j)
            sum -= aug(i, j) * x(j);
        x(i) = sum / aug(i, i);
    }
    return x;
}
```
### Solve Least Squares System
```cpp
Vector SolveLeastSquares() const {
    Matrix Aplus = mpA->pseudoInverse();
    return Aplus * (*mpb);
}
```
