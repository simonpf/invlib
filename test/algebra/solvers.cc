#define BOOST_TEST_MODULE algebra solver
#include <boost/test/included/unit_test.hpp>
#include "algebra.h"
#include "algebra/solvers.h"
#include "algebra/Eigen.h"
#include "utility.h"
#include "test_types.h"

constexpr double EPS = 1e-10;
constexpr unsigned int ntests = 1000;

// Test solvers by computing A * inv(A) * v for a random vector v and a
// random positive definite matrix A. The resulting vector should be equal
// to v up to the precision of the underlying solver.
template
<
typename Matrix
>
void solver_test(unsigned int n)
{

    using Vector = typename Matrix::VectorBase;

    auto A  = random_positive_definite<Matrix>(n);
    auto v = random<Vector>(n);
    Vector w; w.resize(n);

    Standard std{};
    ConjugateGradient cg(1e-12);

    w = A * std.solve(A, v);
    double error = maximum_error(v, w);
    BOOST_TEST((error < EPS), "Standard solver error: " << error);

    w = A * cg.solve(A, v);
    error = maximum_error(v, w);
    BOOST_TEST((error < EPS), "CG solver error: " << error);
}
BOOST_AUTO_TEST_CASE_TEMPLATE(solver,
                              T,
                              matrix_types)
{
    srand(time(NULL));
    for (unsigned int i = 0; i < ntests; i++)
    {
        unsigned int n = 1 + rand() % 100;
        solver_test<T>(n);
    }
}
