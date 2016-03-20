#define BOOST_TEST_MODULE algebra solver
#include <boost/test/included/unit_test.hpp>
#include "algebra.h"
#include "algebra/solvers.h"
#include "utility.h"
#include "test_types.h"

namespace invlib
{

constexpr double EPS = 1e-10;
constexpr unsigned int ntests = 1000;

// Test solvers by computing A * inv(A) * v for a random vector v and a
// random positive definite matrix A. The resulting vector should be equal
// to v up to the precision of the underlying solver.
template
<
typename MatrixType
>
void solver_test(unsigned int n)
{

    using VectorType = typename MatrixType::VectorType;

    auto A  = random_positive_definite<MatrixType>(n);
    auto v = random<VectorType>(n);
    VectorType w; w.resize(n);

    Standard std{};
    ConjugateGradient cg(1e-20);

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

}
