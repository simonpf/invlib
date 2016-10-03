// Copyright (C) 2016 Simon Pfreundschuh - All Rights Reserved
// For licensing term see LICENSE file in the root of this source tree.

/////////////////////////////////////////////////////////////////
// Test different MAP formulations using linear foward models. //
/////////////////////////////////////////////////////////////////

#ifndef BOOST_TEST_MODULE
#define BOOST_TEST_MODULE "Forward Models, Linear"
#endif

#include <boost/test/included/unit_test.hpp>
#include <iostream>

#include "invlib/algebra.h"
#include "invlib/algebra/preconditioners.h"
#include "invlib/algebra/precision_matrix.h"
#include "invlib/algebra/solvers.h"
#include "invlib/map.h"
#include "invlib/optimization.h"

#include "forward_models/linear.h"
#include "utility.h"
#include "test_types.h"

using namespace invlib;

/*
 * Use a random linear forward model to test the equivalence of the
 * standard, n-form and m-form when using the Gauss-Newton optimizer
 * and the standard form using Levenberg-Marquardt and Gauss-Newton
 * optimization.
 */
template
<
typename T
>
void linear_test(unsigned int n)
{
    using RealType   = typename T::RealType;
    using VectorType = typename T::VectorType;
    using MatrixType = typename T::MatrixType;
    using PrecisionMatrix = PrecisionMatrix<MatrixType>;
    using Id     = MatrixIdentity<MatrixType>;
    using Model  = Linear<MatrixType>;
    using Preconditioner = JacobianPreconditioner<VectorType>;

    MatrixType Se = random_positive_definite<MatrixType>(n);
    MatrixType Sa = random_positive_definite<MatrixType>(n);
    MatrixType SeInv = inv(Se);
    MatrixType SaInv = inv(Sa);
    VectorType xa = random<VectorType>(n);
    VectorType y  = random<VectorType>(n);

    PrecisionMatrix Pe(SeInv), Pa(SaInv);

    Model F(n,n);
    MAP<Model, MatrixType, PrecisionMatrix, PrecisionMatrix, VectorType, Formulation::STANDARD>
        std(F, xa, Pa, Pe);
    MAP<Model, MatrixType, PrecisionMatrix, PrecisionMatrix, VectorType, Formulation::NFORM>
        nform(F, xa, Pa, Pe);
    MAP<Model, MatrixType, MatrixType, MatrixType, VectorType, Formulation::MFORM>
        mform(F, xa, Sa, Se);

    // Test inversion using standard solver.
    Id I{};
    GaussNewton<RealType> gn{};
    gn.set_tolerance(1e-9); gn.set_maximum_iterations(1000);
    LevenbergMarquardt<RealType, Id> lm(I);
    lm.set_tolerance(1e-9); lm.set_maximum_iterations(1000);

    VectorType x_std_lm, x_std_gn, x_n_gn, x_m_gn;
    std.compute(x_std_lm, y, lm);
    std.compute(x_std_gn, y, gn);
    nform.compute(x_n_gn, y, gn);
    mform.compute(x_m_gn, y, gn);

    RealType e1, e2, e3;
    e1 = maximum_error(x_std_lm, x_std_gn);
    e2 = maximum_error(x_std_gn, x_n_gn);
    e3 = maximum_error(x_std_gn, x_m_gn);

    BOOST_TEST((e1 < EPS), "Error STD - NFORM = " << e1);
    BOOST_TEST((e2 < EPS), "Error STD - MFORM = " << e2);
    BOOST_TEST((e3 < EPS), "Error STD - MFORM = " << e3);

    // Test inversion using CG solver.
    ConjugateGradient cg(1e-6);
    GaussNewton<RealType, ConjugateGradient> gn_cg(cg);
    gn_cg.set_tolerance(1e-6); gn_cg.set_maximum_iterations(1000);
    LevenbergMarquardt<RealType, Id, ConjugateGradient> lm_cg(I, cg);
    lm_cg.set_tolerance(1e-6); lm_cg.set_maximum_iterations(1000);

    std.compute(x_std_lm, y, lm_cg);
    std.compute(x_std_gn, y, gn_cg);
    nform.compute(x_n_gn, y, gn_cg);
    mform.compute(x_m_gn, y, gn_cg);

    e1 = maximum_error(x_std_lm, x_std_gn);
    e2 = maximum_error(x_std_gn, x_n_gn);
    e3 = maximum_error(x_std_gn, x_m_gn);

    BOOST_TEST((e1 < EPS), "Error STD - NFORM CG = " << e1);
    BOOST_TEST((e2 < EPS), "Error STD - MFORM CG = " << e2);
    BOOST_TEST((e3 < EPS), "Error STD - MFORM CG = " << e3);

    // Test inversion using Preconditioned CG solver.
    using CGType = PreconditionedConjugateGradient<Preconditioner, false>;
    CGType pre_cg(1e-6);
    GaussNewton<RealType, CGType> gn_pre_cg(pre_cg);
    gn_pre_cg.set_tolerance(1e-6); gn_cg.set_maximum_iterations(1000);
    LevenbergMarquardt<RealType, Id, CGType> lm_pre_cg(I, pre_cg);
    lm_pre_cg.set_tolerance(1e-6); lm_cg.set_maximum_iterations(1000);

    std.compute(x_std_lm, y, lm_pre_cg);
    std.compute(x_std_gn, y, gn_pre_cg);
    nform.compute(x_n_gn, y, gn_pre_cg);
    mform.compute(x_m_gn, y, gn_pre_cg);

    e1 = maximum_error(x_std_lm, x_std_gn);
    e2 = maximum_error(x_std_gn, x_n_gn);
    e3 = maximum_error(x_std_gn, x_m_gn);

    BOOST_TEST((e1 < EPS), "Error STD - NFORM PRE CG = " << e1);
    BOOST_TEST((e2 < EPS), "Error STD - MFORM PRE CG = " << e2);
    BOOST_TEST((e3 < EPS), "Error STD - MFORM PRE CG = " << e3);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(linear, T, matrix_types)
{
    srand(time(NULL));
    for (unsigned int i = 0; i < ntests; i++)
    {
        unsigned int n = 1 + rand() % 50;
        linear_test<T>(n);
    }
}
