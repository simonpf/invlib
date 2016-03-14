#define BOOST_TEST_MODULE optimization exact
#include <boost/test/included/unit_test.hpp>
#include "algebra.h"
#include "algebra/solvers.h"
#include "optimization.h"
#include "optimization/test_functions.h"
#include "utility.h"
#include "test_types.h"
#include <iostream>

using namespace invlib;

constexpr double EPS = 1e-10;
constexpr int ntests = 100;

// Tests Gauss-Newton and Levenberg-Marquardt (with lambda = 0) method to yield
// an exact result after one step in the case of a quadratic cost function.
// Uses the sphere functions implemented in the SphereFunction class.
template
<
typename T
>
void exact_minimization(unsigned int n)
{
    using Real   = typename T::Real;
    using Vector = typename T::VectorBase;
    using Matrix = typename T:: MatrixBase;
    using I      = typename Matrix::I;

    Vector x0 = random<Vector>(n);
    Vector dx; dx.resize(n);

    SphereFunction<Real, Vector, Matrix> J(n);

    // Using standard solver.

    I D{};
    LevenbergMarquardt<Real, typename Matrix::I> LM(D);
    LM.lambda_start() = 0.0;
    GaussNewton<Real> GN{};

    auto g = J.gradient(x0);
    auto H = J.Hessian(x0);

    LM.step(dx, x0, g, H, J);
    Vector x = x0 + dx;
    BOOST_TEST((x.norm() / n < EPS), "|x| = " << x.norm());

    GN.step(dx, x0, g, H, J);
    x = x0 + dx;
    BOOST_TEST((x.norm() / n < EPS), "|x| = " << x.norm());

    // Using CG solver.

    ConjugateGradient cg(EPS);
    LevenbergMarquardt<Real, typename Matrix::I, ConjugateGradient> LM_CG(D, cg);
    LM_CG.lambda_start() = 0.0;
    GaussNewton<Real, ConjugateGradient> GN_CG(cg);

    LM_CG.step(dx, x0, g, H, J);
    x = x0 + dx;
    BOOST_TEST((x.norm() / n < EPS), "|x| = " << x.norm());
    std::cout << x << std::endl;

    GN_CG.step(dx, x0, g, H, J);
    x = x0 + dx;
    BOOST_TEST((x.norm() / n < EPS), "|x| = " << x.norm());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(exact, T, matrix_types)
{
    srand(time(NULL));
    for (unsigned int i = 0; i < ntests; i++)
    {
        unsigned int n = 1 + rand() % 100;
        exact_minimization<T>(n);
    }
}
