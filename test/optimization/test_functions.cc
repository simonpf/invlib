#define BOOST_TEST_MODULE optimization test_functions
#include <boost/test/included/unit_test.hpp>
#include "algebra.h"
#include "optimization.h"
#include "optimization/test_functions.h"
#include "utility.h"
#include "test_types.h"
#include <iostream>

using namespace invlib;

constexpr double EPS = 1e-10;
constexpr int ntests = 100;

// Test non-linear optimization using the RandomPowerFunction test function
// which has a minimum at zero and uses the current function value as abortion
// criterion. After the minimization the value of the cost function should be
// lower than the chosen convergence tolerance.
template
<
typename T
>
void random_powers_test(unsigned int n)
{
    using Real   = typename T::Real;
    using Vector = typename T::VectorBase;
    using Matrix = typename T:: MatrixBase;
    using I      = typename Matrix::I;

    Vector x0 = random<Vector>(n);
    Vector dx; dx.resize(n);

    RandomPowerFunction<Real, Vector, Matrix> J(n);
    I D{};
    LevenbergMarquardt<Real, typename Matrix::I> LM(D);
    GaussNewton<Real> GN{};

    Vector x;
    minimize(J, LM, x0, x, 1000, EPS);
    BOOST_TEST((J.cost_function(x) < EPS), "J(x) = " << J.cost_function(x));

    minimize(J, GN, x0, x, 1000, EPS);
    BOOST_TEST((J.cost_function(x) < EPS), "J(x) = " << J.cost_function(x));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(random_powers, T, matrix_types)
{
    srand(time(NULL));
    for (unsigned int i = 0; i < ntests; i++)
    {
        unsigned int n = 1 + rand() % 100;
        random_powers_test<T>(n);
    }
}
