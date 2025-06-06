#include <iostream>
#include <cmath>

#include "Matrix.cpp"

class LinearSystem {
protected:
    int     mSize;   // dimension if A is square; otherwise 0
    Matrix *mpA;     // pointer to coefficient matrix
    Vector *mpb;     // pointer to right-hand side vector

public:
    // Delete default and copy constructors
    LinearSystem() = delete;
    LinearSystem(const LinearSystem &other) = delete;

    // Constructor: copy A and b
    LinearSystem(const Matrix &A, const Vector &b)
        : mpA(nullptr), mpb(nullptr), mSize(0)
    {
        int rows = A.numRows();
        int cols = A.numCols();
        if(b.size() != rows) {
            std::cerr << "Dimension mismatch in LinearSystem constructor.\n";
            std::exit(EXIT_FAILURE);
        }
        mpA = new Matrix(A);
        mpb = new Vector(b);
        if(rows == cols) {
            mSize = rows;
        } else {
            mSize = 0;
        }
    }

    // Destructor
    virtual ~LinearSystem()
    {
        delete mpA;
        delete mpb;
    }

    // Solve Ax = b by Gaussian elimination (square only)
    virtual Vector Solve() const
    {
        if(mSize <= 0) {
            std::cerr << "Solve() requires a square system.\n";
            std::exit(EXIT_FAILURE);
        }
        int n = mSize;
        Matrix aug(n, n + 1);
        for(int i = 1; i <= n; ++i) {
            for(int j = 1; j <= n; ++j) {
                aug(i, j) = (*mpA)(i, j);
            }
            aug(i, n + 1) = (*mpb)(i);
        }
        // Forward elimination with partial pivoting
        for(int i = 1; i <= n; ++i) {
            int pivot = i;
            double maxAbs = std::fabs(aug(i, i));
            for(int row = i + 1; row <= n; ++row) {
                double val = std::fabs(aug(row, i));
                if(val > maxAbs) {
                    maxAbs = val;
                    pivot = row;
                }
            }
            if(std::fabs(aug(pivot, i)) < 1e-12) {
                std::cerr << "Matrix is singular or nearly singular.\n";
                std::exit(EXIT_FAILURE);
            }
            if(pivot != i) {
                for(int col = i; col <= n + 1; ++col) {
                    std::swap(aug(i, col), aug(pivot, col));
                }
            }
            for(int row = i + 1; row <= n; ++row) {
                double factor = aug(row, i) / aug(i, i);
                for(int col = i; col <= n + 1; ++col) {
                    aug(row, col) -= factor * aug(i, col);
                }
            }
        }
        // Back substitution
        Vector x(n);
        for(int i = n; i >= 1; --i) {
            double sum = aug(i, n + 1);
            for(int j = i + 1; j <= n; ++j) {
                sum -= aug(i, j) * x(j);
            }
            x(i) = sum / aug(i, i);
        }
        return x;
    }

    // Solve least squares x = A^+ b
    Vector SolveLeastSquares() const
    {
        Matrix Aplus = mpA->pseudoInverse();       // size n×m if A is m×n
        Vector x = Aplus * (*mpb);
        return x;
    }

    // Solve (A^T A + λI)x = A^T b
    Vector SolveRegularized(double lambda) const
    {
        int m = mpA->numRows();
        int n = mpA->numCols();
        Matrix At = mpA->transpose();              // n×m
        Matrix AtA = At * (*mpA);                  // n×n
        for(int i = 1; i <= n; ++i) {
            AtA(i, i) += lambda;
        }
        Vector Atb(n);
        for(int i = 1; i <= n; ++i) {
            double sum = 0.0;
            for(int j = 1; j <= m; ++j) {
                sum += At(i, j) * (*mpb)(j);
            }
            Atb(i) = sum;
        }
        LinearSystem normalSys(AtA, Atb);
        Vector x = normalSys.Solve();
        return x;
    }
};