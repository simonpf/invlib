#ifndef BOOST_TEST_MODULE
#define BOOST_TEST_MODULE "Algebra, Solvers"
#endif

#include <boost/test/included/unit_test.hpp>
#include "invlib/algebra.h"
#include "invlib/algebra/solvers.h"
#include "invlib/algebra/preconditioners.h"
#include "utility.h"
#include "test_types.h"

using namespace invlib;

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

void cg_test(unsigned int n)
{
    using MatrixType = Matrix<MatrixArchetype<double>>;
    using VectorType = Vector<VectorArchetype<double>>;
    using Preconditioner = JacobianPreconditioner<VectorType>;

    auto A  = random_positive_definite<MatrixType>(n);
    auto v = random<VectorType>(n);
    VectorType w; w.resize(n);

    Standard std{};
    JacobianPreconditioner<VectorType> pre(A);
    PreconditionedConjugateGradient<Preconditioner, true> cg(pre, 1e-20);

    w = A * std.solve(A, v);
    double error = maximum_error(v, w);
    BOOST_TEST((error < EPS), "CG solver error:                " << error);

    w = A * cg.solve(A, v);
    error = maximum_error(v, w);
    BOOST_TEST((error < EPS), "Preconditioned CG solver error: " << error);
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

BOOST_AUTO_TEST_CASE(preconditioned_cg)
{
    srand(time(NULL));
    for (unsigned int i = 0; i < ntests; i++)
    {
        unsigned int n = 1 + rand() % 100;
        cg_test(n);
    }
}
