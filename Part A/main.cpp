#include <iostream>

#include "Vector.cpp"
#include "Matrix.cpp"
#include "LinearSystem.cpp"
#include "PosSymLinSystem.cpp"

int main()
{
    // Example 1: Solve a simple 3×3 system by Gaussian elimination
    {
        int n = 3;
        Matrix A(n, n);
        A(1,1) = 4;  A(1,2) = 1;  A(1,3) = 2;
        A(2,1) = 1;  A(2,2) = 5;  A(2,3) = 3;
        A(3,1) = 2;  A(3,2) = 3;  A(3,3) = 10;

        Vector b(n);
        b(1) = 7;  b(2) = 12;  b(3) = 20;

        LinearSystem sys(A, b);
        Vector x = sys.Solve();
        std::cout << "Solution of A x = b via Gaussian elimination:\n";
        for(int i = 1; i <= n; ++i) {
            std::cout << "  x[" << i << "] = " << x(i) << "\n";
        }
        std::cout << std::endl;
    }

    // Example 2: Solve same SPD system via Conjugate Gradient
    {
        int n = 3;
        Matrix A(n, n);
        A(1,1) = 4;  A(1,2) = 1;  A(1,3) = 2;
        A(2,1) = 1;  A(2,2) = 5;  A(2,3) = 3;
        A(3,1) = 2;  A(3,2) = 3;  A(3,3) = 10;

        Vector b(n);
        b(1) = 7;  b(2) = 12;  b(3) = 20;

        PosSymLinSystem cgSys(A, b);
        Vector xCG = cgSys.Solve();
        std::cout << "Solution of SPD system via Conjugate Gradient:\n";
        for(int i = 1; i <= n; ++i) {
            std::cout << "  x_CG[" << i << "] = " << xCG(i) << "\n";
        }
        std::cout << std::endl;
    }

    // Example 3: Under‐determined: m = 2, n = 3, A x = b (least‐squares)
    {
        Matrix A(2, 3);
        A(1,1) = 1;  A(1,2) = 2;  A(1,3) = 3;
        A(2,1) = 4;  A(2,2) = 5;  A(2,3) = 6;

        Vector b(2);
        b(1) = 14;  b(2) = 32;

        LinearSystem ls(A, b);
        Vector xLS = ls.SolveLeastSquares();
        std::cout << "Least‐squares solution x = A^+ b (A is 2×3):\n";
        for(int i = 1; i <= 3; ++i) {
            std::cout << "  x_LS[" << i << "] = " << xLS(i) << "\n";
        }
        std::cout << std::endl;

        double lambda = 0.1;
        Vector xTik = ls.SolveRegularized(lambda);
        std::cout << "Tikhonov‐regularized solution (λ = " << lambda << "):\n";
        for(int i = 1; i <= 3; ++i) {
            std::cout << "  x_Tik[" << i << "] = " << xTik(i) << "\n";
        }
        std::cout << std::endl;
    }

    return 0;
}