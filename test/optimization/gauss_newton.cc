#include <iostream>

#include "algebra.h"
#include "algebra/Eigen.h"
#include "optimization.h"
#include "test_functions.h"
#include "../utility.h"

using std::cout;
using std::endl;

template <typename T> void foo(T a);

int main( int argc, const char **argv )
{

    typedef SphereFunction<double, EigenVector, EigenMatrix> CostFunction;
    typedef LevenbergMarquardt<double, I> LM;
    typedef GaussNewton<double> GN;

    unsigned int n = 10;
    EigenVector v,w;
    v.resize(10); w.resize(10);

    CostFunction S(n);
    I D{};
    LM Minimizer(D);
    //SumOfPowers< double, Matrix<double>, Vector<double> > S(n);

    v = random<EigenVector>(n);
    cout << v << endl;

    minimize(S, Minimizer, v, w, 10, 1e-12);

    cout << w << endl;
}
