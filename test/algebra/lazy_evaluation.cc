#include "algebra/Eigen.h"
#include <stdio.h>

int main( int argc, const char** argv )
{

    Matrix<double> A(1000,1000), B(1000,1000), C(1000,1000), D(1000,1000),
        E(100,100), F(100,100), G(100,100);

    A(0,0) = 1;
    B(0,0) = 1;
    C(0,0) = 1;

    printf("product\n");
    D = A * B * C;
    printf("product\n");

    printf("D(0,0) = %f \n", G(0,0));
}
