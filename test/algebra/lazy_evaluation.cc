#include "algebra/Eigen.h"
#include <stdio.h>

int main( int argc, const char** argv )
{

    Matrix<double> A(1000,1000), B(1000,1000), C(1000,1000), D(1000,1000),
        E(1000,1000), F(1000,1000), G(1000,1000), H(1000,1000), I;

    A(0,0) = 1;
    B(0,0) = 1;
    C(0,0) = 1;

    printf("product\n");
    I = A * B * C * D * E * F * G * H;
    printf("product\n");

    printf("D(0,0) = %f \n", G(0,0));
}
