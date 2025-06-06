#include <iostream>
#include <cmath>

#include "LinearSystem.cpp"  // for LinearSystem, Matrix, Vector

class PosSymLinSystem : public LinearSystem {
public:
    // Constructor: forward to base
    PosSymLinSystem(const Matrix &A, const Vector &b)
        : LinearSystem(A, b)
    {
        int n = A.numRows();
        if(n != A.numCols()) {
            std::cerr << "Matrix must be square for PosSymLinSystem.\n";
            std::exit(EXIT_FAILURE);
        }
        // Check symmetry (no assert)
        for(int i = 1; i <= n; ++i) {
            for(int j = i + 1; j <= n; ++j) {
                double aij = A(i, j);
                double aji = A(j, i);
                if(std::fabs(aij - aji) > 1e-9) {
                    std::cerr << "Matrix is not symmetric.\n";
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }

    // Override Solve() to use Conjugate Gradient
    virtual Vector Solve() const override
    {
        int n = mSize;
        if(n <= 0) {
            std::cerr << "Solve() requires a square system.\n";
            std::exit(EXIT_FAILURE);
        }

        const Matrix &A = *mpA;
        const Vector &b = *mpb;

        Vector x(n);
        for(int i = 1; i <= n; ++i) {
            x(i) = 0.0;
        }

        Vector r = b;      // r0 = b - A x0 = b
        Vector p = r;      // p0 = r0
        double rsold = r.dot(r);
        const double tol = 1e-9;
        Vector Ap(n);

        for(int k = 0; k < n; ++k) {
            Ap = A * p;                          // Ap = A * p
            double alpha = rsold / p.dot(Ap);
            for(int i = 1; i <= n; ++i) {
                x(i) += alpha * p(i);
            }
            for(int i = 1; i <= n; ++i) {
                r(i) -= alpha * Ap(i);
            }
            double rsnew = r.dot(r);
            if(std::sqrt(rsnew) < tol) {
                break;
            }
            double beta = rsnew / rsold;
            for(int i = 1; i <= n; ++i) {
                p(i) = r(i) + beta * p(i);
            }
            rsold = rsnew;
        }
        return x;
    }
};