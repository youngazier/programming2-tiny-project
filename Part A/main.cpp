#include <iostream>
#include "Vector.h"
#include "Matrix.h"
#include "LinearSystem.h"

using namespace std;

int main() {
    // Test 1: Solve Ax = b using Gaussian Elimination
    cout << "=== LinearSystem: Gaussian Elimination ===" << endl;
    Matrix A(3, 3);
    A(1,1) = 2; A(1,2) = -1; A(1,3) = 0;
    A(2,1) = -1; A(2,2) = 2; A(2,3) = -1;
    A(3,1) = 0; A(3,2) = -1; A(3,3) = 2;

    Vector b(3);
    b[0] = 1; b[1] = 0; b[2] = 1;

    LinearSystem sys(A, b);
    Vector x = sys.Solve();

    cout << "Solution x:" << endl;
    x.print();

    // Test 2: Solve symmetric system using Conjugate Gradient
    cout << "\n=== PosSymLinSystem: Conjugate Gradient ===" << endl;
    Matrix S(3, 3);
    S(1,1) = 4; S(1,2) = 1; S(1,3) = 1;
    S(2,1) = 1; S(2,2) = 3; S(2,3) = 0;
    S(3,1) = 1; S(3,2) = 0; S(3,3) = 2;

    Vector b2(3);
    b2[0] = 1; b2[1] = 2; b2[2] = 3;

    PosSymLinSystem psys(S, b2);
    Vector x2 = psys.Solve();

    cout << "Solution x:" << endl;
    x2.print();

    return 0;
}