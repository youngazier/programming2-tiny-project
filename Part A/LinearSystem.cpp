#include "LinearSystem.h"
#include <cassert>
#include <cmath>

LinearSystem::LinearSystem(const Matrix& A, const Vector& b) {
    assert(A.numRows() == A.numCols() && A.numRows() == b.size());
    mSize = A.numRows();
    mpA = new Matrix(A);
    mpb = new Vector(b);
}

LinearSystem::~LinearSystem() {
    delete mpA;
    delete mpb;
}

Vector LinearSystem::Solve() {
    Matrix A = *mpA;
    Vector b = *mpb;
    int n = mSize;

    for (int i = 0; i < n; ++i) {
        // Pivot
        int maxRow = i;
        for (int k = i + 1; k < n; ++k) {
            if (fabs(A(k + 1, i + 1)) > fabs(A(maxRow + 1, i + 1)))
                maxRow = k;
        }
        for (int j = 0; j < n; ++j)
            std::swap(A(i + 1, j + 1), A(maxRow + 1, j + 1));
        std::swap(b[i], b[maxRow]);

        // Eliminate
        for (int k = i + 1; k < n; ++k) {
            double factor = A(k + 1, i + 1) / A(i + 1, i + 1);
            for (int j = i; j < n; ++j)
                A(k + 1, j + 1) -= factor * A(i + 1, j + 1);
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    Vector x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i + 1; j < n; ++j)
            x[i] -= A(i + 1, j + 1) * x[j];
        x[i] /= A(i + 1, i + 1);
    }
    return x;
}

PosSymLinSystem::PosSymLinSystem(const Matrix& A, const Vector& b) : LinearSystem(A, b) {
    assert(A.numRows() == A.numCols());
    for (int i = 1; i <= A.numRows(); ++i)
        for (int j = 1; j <= A.numCols(); ++j)
            assert(fabs(A(i, j) - A(j, i)) < 1e-9); // check symmetry
}

Vector PosSymLinSystem::Solve() {
    int n = mSize;
    Vector x(n), r = *mpb - (*mpA) * x;
    Vector p = r;

    for (int k = 0; k < n; ++k) {
        Vector Ap = (*mpA) * p;
        double alpha = (r * r) / (p * Ap);
        x = x + p * alpha;
        Vector r_new = r - Ap * alpha;
        if ((r_new * r_new) < 1e-10) break;
        double beta = (r_new * r_new) / (r * r);
        p = r_new + p * beta;
        r = r_new;
    }
    return x;
}
