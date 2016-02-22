#include <iostream>

#include "algebra/Eigen.h"
#include "algebra/matrix_identity.h"
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

    typedef IdentityMatrix<double, Matrix> Identity;
    typedef SphereFunction<double, Vector, Matrix> CostFunction;
    typedef LevenbergMarquardt<double, Identity> Minimizer;

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
