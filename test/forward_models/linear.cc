#define BOOST_TEST_MODULE forward_models linear
#include <boost/test/included/unit_test.hpp>
#include "algebra.h"
#include "map.h"
#include "optimization.h"
#include "forward_models/linear.h"
#include "utility.h"
#include "test_types.h"
#include <iostream>

constexpr double EPS = 1e-9;
constexpr int ntests = 100;

// Use a random linear forward model to test the equivalence of the
// standard, n-form and m-form when using the Gauss-Newton optimizer
// and the standard form using Levenberg-Marquardt and Gauss-Newton
// optimization.
template
<
typename T
>
void linear_test(unsigned int n)
{
    using Real   = typename T::Real;
    using Vector = typename T::VectorBase;
    using Matrix = typename T:: MatrixBase;
    using I      = typename Matrix::I;
    using Model  = Linear<Matrix>;

    Matrix Se = random_positive_definite<Matrix>(n);
    Matrix Sa = random_positive_definite<Matrix>(n);
    Vector xa = random<Vector>(n);
    Vector y  = random<Vector>(n);

    Model F(n,n);
    MAP<Model, Real, Vector, Matrix, Matrix, Matrix, Formulation::STANDARD>
        std(F, xa, Sa, Se);
    MAP<Model, Real, Vector, Matrix, Matrix, Matrix, Formulation::NFORM>
        nform(F, xa, Sa, Se);
    MAP<Model, Real, Vector, Matrix, Matrix, Matrix, Formulation::MFORM>
        mform(F, xa, Sa, Se);

    GaussNewton<Real> GN{};
    GN.tolerance() = 1e-9; GN.maximum_iterations() = 1000;
    I Id{};
    LevenbergMarquardt<Real, I> LM(Id);
    LM.tolerance() = 1e-9; LM.maximum_iterations() = 1000;

    Vector x_std_lm, x_std_gn, x_n_gn, x_m_gn;
    std.compute(x_std_lm, y, LM);
    std.compute(x_std_gn, y, GN);
    nform.compute(x_n_gn, y, GN);
    mform.compute(x_m_gn, y, GN);

    Real e1, e2, e3;
    e1 = maximum_error(x_std_lm, x_std_gn);
    e2 = maximum_error(x_std_gn, x_n_gn);
    e3 = maximum_error(x_std_gn, x_m_gn);

    BOOST_TEST((e1 < EPS), "Error STD - NFORM = " << e1);
    BOOST_TEST((e2 < EPS), "Error STD - MFORM = " << e2);
    BOOST_TEST((e3 < EPS), "Error STD - MFORM = " << e3);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(linear, T, matrix_types)
{
    srand(time(NULL));
    for (unsigned int i = 0; i < ntests; i++)
    {
        unsigned int n = 1 + rand() % 100;
        linear_test<T>(n);
    }
}
