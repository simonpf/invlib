#define BOOST_TEST_MODULE forward_models sphere
#include <boost/test/included/unit_test.hpp>
#include "algebra.h"
#include "map.h"
#include "optimization.h"
#include "forward_models/sphere.h"
#include "utility.h"
#include "test_types.h"
#include <iostream>

constexpr double EPS = 1e-9;
constexpr int ntests = 100;

// Use the spehere function forward model to test the equivalence of the
// standard, n-form and m-form when using the Gauss-Newton optimizer.
template
<
typename T
>
void sphere_test(unsigned int n)
{
    using Real   = typename T::Real;
    using Vector = typename T::VectorBase;
    using Matrix = typename T:: MatrixBase;
    using Model  = Sphere<Matrix>;

    Matrix Se = random_positive_definite<Matrix>(1);
    Matrix Sa = random_positive_definite<Matrix>(n);
    Vector xa = random<Vector>(n);
    Vector y  = random<Vector>(1);

    Model F(n);
    MAP<Model, Real, Vector, Matrix, Matrix, Matrix, Formulation::STANDARD>
        std(F, xa, Sa, Se);
    MAP<Model, Real, Vector, Matrix, Matrix, Matrix, Formulation::NFORM>
        nform(F, xa, Sa, Se);
    MAP<Model, Real, Vector, Matrix, Matrix, Matrix, Formulation::MFORM>
        mform(F, xa, Sa, Se);

    GaussNewton<Real> GN{};
    GN.tolerance() = 1e-40; GN.maximum_iterations() = 1000;

    Vector x_std, x_n, x_m;
    std.compute(x_std, y, GN);
    nform.compute(x_n, y, GN);
    mform.compute(x_m, y, GN);

    Real e1, e2;
    e1 = maximum_error(x_std, x_m);
    e2 = maximum_error(x_std, x_n);

    BOOST_TEST((e1 < EPS), "Error STD - NFORM = " << e1);
    BOOST_TEST((e2 < EPS), "Error STD - MFORM =" << e2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sphere, T, matrix_types)
{
    srand(time(NULL));
    for (unsigned int i = 0; i < ntests; i++)
    {
        unsigned int n = 1 + rand() % 100;
        sphere_test<T>(n);
    }
}
