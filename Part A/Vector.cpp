#pragma once
#include <iostream>

class Vector {
private:
    int    mSize;   // number of elements
    double *mData;  // pointer to dynamically allocated array

public:
    // Constructor: allocate array of given size and initialize to zero
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

    // Copy constructor: deep copy
    Vector(const Vector &other)
        : mSize(other.mSize), mData(nullptr)
    {
        mData = new double[mSize];
        for(int i = 0; i < mSize; ++i) {
            mData[i] = other.mData[i];
        }
    }

    // Destructor: free memory
    ~Vector()
    {
        delete[] mData;
    }

    // Assignment operator: deep copy
    Vector& operator=(const Vector &other)
    {
        if(this == &other) return *this;
        if(mSize != other.mSize) {
            delete[] mData;
            mSize = other.mSize;
            mData = new double[mSize];
        }
        for(int i = 0; i < mSize; ++i) {
            mData[i] = other.mData[i];
        }
        return *this;
    }

    // Return number of elements
    int size() const
    {
        return mSize;
    }

    // Zero‐based indexing (no bound checks)
    double& operator[](int index)
    {
        return mData[index];
    }
    const double& operator[](int index) const
    {
        return mData[index];
    }

    // One‐based indexing (no bound checks)
    double& operator()(int i)
    {
        return mData[i - 1];
    }
    const double& operator()(int i) const
    {
        return mData[i - 1];
    }

    // Vector addition
    Vector operator+(const Vector &other) const
    {
        // Dimensions should match; omitted check
        Vector result(mSize);
        for(int i = 0; i < mSize; ++i) {
            result.mData[i] = mData[i] + other.mData[i];
        }
        return result;
    }

    // Vector subtraction
    Vector operator-(const Vector &other) const
    {
        Vector result(mSize);
        for(int i = 0; i < mSize; ++i) {
            result.mData[i] = mData[i] - other.mData[i];
        }
        return result;
    }

    // Scalar multiplication (vector * scalar)
    Vector operator*(double scalar) const
    {
        Vector result(mSize);
        for(int i = 0; i < mSize; ++i) {
            result.mData[i] = mData[i] * scalar;
        }
        return result;
    }

    // Dot product
    double dot(const Vector &other) const
    {
        double sum = 0.0;
        for(int i = 0; i < mSize; ++i) {
            sum += mData[i] * other.mData[i];
        }
        return sum;
    }

    // Friend: scalar * vector
    friend Vector operator*(double scalar, const Vector &vec)
    {
        return vec * scalar;
    }
};
