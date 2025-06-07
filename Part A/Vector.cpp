#include "Vector.h"
#include <iostream>
using namespace std;

Vector::Vector(int size) : mSize(size) {
    mData = new double[mSize];
    for (int i = 0; i < mSize; ++i)
        mData[i] = 0.0;
}

Vector::Vector(const Vector& other) : mSize(other.mSize) {
    mData = new double[mSize];
    for (int i = 0; i < mSize; ++i)
        mData[i] = other.mData[i];
}

Vector::~Vector() {
    delete[] mData;
}

Vector& Vector::operator=(const Vector& other) {
    if (this != &other) {
        delete[] mData;
        mSize = other.mSize;
        mData = new double[mSize];
        for (int i = 0; i < mSize; ++i)
            mData[i] = other.mData[i];
    }
    return *this;
}

Vector Vector::operator+(const Vector& other) const {
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i)
        result.mData[i] = mData[i] + other.mData[i];
    return result;
}

Vector Vector::operator-(const Vector& other) const {
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i)
        result.mData[i] = mData[i] - other.mData[i];
    return result;
}

Vector Vector::operator*(double scalar) const {
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i)
        result.mData[i] = mData[i] * scalar;
    return result;
}

double Vector::operator*(const Vector& other) const {
    double dot = 0.0;
    for (int i = 0; i < mSize; ++i)
        dot += mData[i] * other.mData[i];
    return dot;
}

double& Vector::operator[](int index) {
    return mData[index]; // 0-based index
}

double Vector::operator()(int index) const {
    return mData[index - 1]; // 1-based index
}

int Vector::size() const {
    return mSize;
}

void Vector::print() const {
    for (int i = 0; i < mSize; ++i)
        cout << mData[i] << " ";
    cout << endl;
}
