#include "map.h"
#include "../test/utility.h"
#include "../forward_models/sphere.h"
#include "algebra.h"
#include "algebra/Eigen.h"
#include "optimization/levenberg_marquardt.h"
#include "optimization/gauss_newton.h"
#include "../optimization/test_functions.h"
#include <cmath>

#include <iostream>

template
<
typename TestFunction,
typename Real,
typename Vector,
typename Matrix
>
Real test_map( unsigned int n,
               unsigned int ntests,
               bool verbose )
{
    typedef LevenbergMarquardt<Real,
                               MatrixIdentity<Real, Matrix>,
                               Verbosity::SILENT> LM;
    typedef GaussNewton<Real> GN;
    typedef MAP<TestFunction, double, Vector, Matrix, MatrixIdentity<Real, Matrix>,
                MatrixIdentity<Real, Matrix>, Formulation::STANDARD> MAP;

    for (unsigned int i = 0; i < n; i++)
    {
       Vector x{}, y{};
        x.resize(n); y.resize(1);
        Vector xa = random<Vector>(n);
        y[0] = dot(xa,xa);

        TestFunction S(n);
        MatrixIdentity<Real, Matrix> I(1.0);
        MatrixZero<Matrix> Zero;

        std::cout << "Performing test " << i << " of " << ntests << std::endl;
        std::cout << "Start vector: " << std::endl << xa << std::endl;

        MAP estimator(S, xa, I, I);
        LM minimizer{I}; minimizer.maximum_iterations(1000);
        estimator.compute(x, y, minimizer);

        std::cout << "Result" << std::endl << x << std::endl;
    }
}


int main(int argc, const char** argv)
{

    typedef Sphere<EigenMatrix> CostFunction;
    test_map<CostFunction, double, EigenVector, EigenMatrix>(10, 10, false);

}
