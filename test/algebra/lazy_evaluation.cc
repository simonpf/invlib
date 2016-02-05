#include "algebra/Eigen.h"
#include <stdio.h>

int main( int argc, const char** argv )
{

    Matrix<double> A(1000,1000), B(1000,1000), C(1000,1000), D(1000,1000),
        E(1000,1000), F(1000,1000), G(1000,1000), H(1000,1000);
    Vector<double> v(1000), w(1000);

    A(0,0) = 1;
    B(0,0) = 1;
    C(0,0) = 1;

    for (int i = 0; i < 1000; i++)
    {
        A(i,i) = 2.0;
        v(i)   = 1.0;
    }

    B = inv(A);
    w =  B*v;
    printf("B(0,0) = %f \n v(0) = %f \n", B(0,0), w(0));
}
