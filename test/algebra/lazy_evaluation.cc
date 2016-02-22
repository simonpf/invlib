#include "algebra/matrix.h"
#include "algebra/vector.h"
#include "algebra/matrix_inverse.h"
#include "algebra/matrix_identity.h"
#include "algebra/matrix_transpose.h"
#include "algebra/Eigen.h"
#include "Eigen/Dense"
#include <stdio.h>

template <typename T> void foo(T a);

int main( int argc, const char** argv )
{

    const unsigned int n = 2;
    using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Vector<Eigen::Matrix<double, Eigen::Dynamic, 1>>;
    Matrix<EigenWrapper, Vector> A, B, C, D, E, F, G, H;
    A.resize(n,n); B.resize(n,n); C.resize(n,n); D.resize(n,n);
    E.resize(n,n); F.resize(n,n); G.resize(n,n); H.resize(n,n);

    using I = MatrixIdentity<double, Matrix<EigenWrapper, Vector> >;

    Vector v, w;
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
    H = transp(A) * B;
    w = (A + B) * inv(C + D) * v;

    printf("H(0,0) = %f \n v(0) = %f \n", H(0,0), w(0));
}
