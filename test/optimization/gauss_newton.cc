#include <iostream>

#include "algebra/Eigen.h"
#include "algebra/identity.h"
#include "test_functions.h"
#include "optimization/gauss_newton.h"
#include "optimization/levenberg_marquardt.h"
#include "optimization/minimize.h"
#include "../utility.h"


using std::cout;
using std::endl;

template <typename T> void foo(T a);

int main( int argc, const char **argv )
{

    typedef SphereFunction< double,
                            Vector<double>,
                            Matrix<double> > CostFunction;
    typedef LevenbergMarquardt< double, Vector, Matrix,
                                SphereFunction, IdentityMatrix > Minimizer;

    unsigned int n = 10;
    Vector<double> v(10),w(10);



    CostFunction S(n);
    Minimizer LM = Minimizer(IdentityMatrix<double,Matrix>());
    //SumOfPowers< double, Matrix<double>, Vector<double> > S(n);

    v = random<Vector<double> > (n);
    cout << v << endl;

    minimize(S, LM, v, w, 10, 1e-12);

    cout << w << endl;
}
