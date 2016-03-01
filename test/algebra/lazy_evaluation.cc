#include "algebra.h"
#include "algebra/solver.h"
#include "algebra/Eigen.h"
#include "utility.h"
#include <stdio.h>
#include <iostream>

template <typename T> void foo(T a);

int main( int argc, const char** argv )
{

    const unsigned int n = 5;
    EigenMatrix A, B, C, D, E, F, G, H;
    MatrixIdentity<double, EigenMatrix> I;
    EigenVector v, w;

    A.resize(n,n); B.resize(n,n); C.resize(n,n); D.resize(n,n);
    E.resize(n,n); F.resize(n,n); G.resize(n,n); H.resize(n,n);

    v.resize(n); w.resize(n);

    A(0,0) = 1;
    A(1,0) = 1;
    B(0,0) = 2;
    B(1,0) = 2;
    C(0,0) = 3;
    D(0,0) = 4;
    E(0,0) = 5;
    F(0,0) = 6;
    G(0,0) = 7;

    for (int i = 0; i < n; i++)
    {
        v(i)   = 1.0;
    }

    set_identity(A);
    A *= 2.0;

    A = random_positive_definite<EigenMatrix>(n);
    ConjugateGradient CG{};
    w = A * CG.solve(A, v);

    std::cout << A << std::endl;
    std::cout << v << std::endl << " --- " << std::endl;
    std::cout << w << std::endl;
    //printf("H(0,0) = %f \n v(0) = %f \n", maximum_error(w,v), w(0));
}
