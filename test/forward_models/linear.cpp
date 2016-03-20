#define BOOST_TEST_MODULE forward_models linear
#include <boost/test/included/unit_test.hpp>
#include "algebra.h"
#include "algebra/solvers.h"
#include "map.h"
#include "optimization.h"
#include "forward_models/linear.h"
#include "utility.h"
#include "test_types.h"
#include <iostream>

using namespace invlib;

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
    using RealType   = typename T::RealType;
    using VectorType = typename T::VectorType;
    using MatrixType = typename T::MatrixType;
    using Id   = MatrixIdentity<MatrixType>;
    using Model  = Linear<MatrixType>;

    MatrixType Se = random_positive_definite<MatrixType>(n);
    MatrixType Sa = random_positive_definite<MatrixType>(n);
    VectorType xa = random<VectorType>(n);
    VectorType y  = random<VectorType>(n);

    Model F(n,n);
    MAP<Model, RealType, VectorType, MatrixType,
        MatrixType, MatrixType, Formulation::STANDARD> std(F, xa, Sa, Se);
    MAP<Model, RealType, VectorType, MatrixType,
        MatrixType, MatrixType, Formulation::NFORM>    nform(F, xa, Sa, Se);
    MAP<Model, RealType, VectorType, MatrixType,
        MatrixType, MatrixType, Formulation::MFORM>    mform(F, xa, Sa, Se);

    // Test inversion using standard solver.
    Id I{};
    GaussNewton<RealType> GN{};
    GN.tolerance() = 1e-9; GN.maximum_iterations() = 1000;
    LevenbergMarquardt<RealType, Id> LM(I);
    LM.tolerance() = 1e-9; LM.maximum_iterations() = 1000;

    VectorType x_std_lm, x_std_gn, x_n_gn, x_m_gn;
    std.compute(x_std_lm, y, LM);
    std.compute(x_std_gn, y, GN);
    nform.compute(x_n_gn, y, GN);
    mform.compute(x_m_gn, y, GN);

    RealType e1, e2, e3;
    e1 = maximum_error(x_std_lm, x_std_gn);
    e2 = maximum_error(x_std_gn, x_n_gn);
    e3 = maximum_error(x_std_gn, x_m_gn);

    BOOST_TEST((e1 < EPS), "Error STD - NFORM = " << e1);
    BOOST_TEST((e2 < EPS), "Error STD - MFORM = " << e2);
    BOOST_TEST((e3 < EPS), "Error STD - MFORM = " << e3);

    // Test inversion using CG solver.
    ConjugateGradient cg(1e-5);
    GaussNewton<RealType, ConjugateGradient> GN_CG(cg);
    GN_CG.tolerance() = 1e-9; GN.maximum_iterations() = 1000;
    LevenbergMarquardt<RealType, Id, ConjugateGradient> LM_CG(I, cg);
    LM_CG.tolerance() = 1e-9; LM.maximum_iterations() = 1000;

    std.compute(x_std_lm, y, LM_CG);
    std.compute(x_std_gn, y, GN_CG);
    nform.compute(x_n_gn, y, GN_CG);
    mform.compute(x_m_gn, y, GN_CG);

    BOOST_TEST((e1 < EPS), "Error STD - NFORM CG = " << e1);
    BOOST_TEST((e2 < EPS), "Error STD - MFORM CG = " << e2);
    BOOST_TEST((e3 < EPS), "Error STD - MFORM CG = " << e3);
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
