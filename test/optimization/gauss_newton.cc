#include <iostream>

#include "algebra/Eigen.h"
#include "test_functions.h"
#include "optimization/gauss_newton.h"
#include "../utility.h"


using std::cout;
using std::endl;

int main( int argc, const char **argv )
{
    unsigned int n = 10;
    Vector<double> v(10),w(10);
    //SphereFunction< double, Matrix<double>, Vector<double> > S(n);
    SumOfPowers< double, Matrix<double>, Vector<double> > S(n);

    v = random<Vector<double> > (n);
    cout << v << endl;

    gauss_newton( S, v, w, 100, 1e-5 );

    cout << w << endl;
}
