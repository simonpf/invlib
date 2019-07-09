// Copyright (C) 2016 Simon Pfreundschuh - All Rights Reserved
// For licensing term see LICENSE file in the root of this source tree.

/////////////////////////////////////////////////////////////////
// Test different MAP formulations using linear foward models. //
/////////////////////////////////////////////////////////////////

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
template <typename T>
struct LinearModel {

    static constexpr char name[] = "Linear";

    static void run(size_t n) {
        using RealType   = typename T::RealType;
        using VectorType = Vector<typename T::VectorType>;
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

        auto e1 = maximum_error(x_std_lm, x_std_gn);
        auto e2 = maximum_error(x_std_gn, x_n_gn);
        auto e3 = maximum_error(x_std_gn, x_m_gn);

        ensure_small(e1, "Error standard LM - standard GN");
        ensure_small(e2, "Error standard GN - n-form GN");
        ensure_small(e3, "Error standard GN - n-form GN");

        // Test inversion using CG solver.
        ConjugateGradient<> cg(1e-12);
        GaussNewton<RealType, ConjugateGradient<>> gn_cg(cg);
        gn_cg.set_tolerance(1e-6); gn_cg.set_maximum_iterations(1000);
        LevenbergMarquardt<RealType, Id, ConjugateGradient<>> lm_cg(I, cg);
        lm_cg.set_tolerance(1e-6); lm_cg.set_maximum_iterations(1000);

        std.compute(x_std_lm, y, lm_cg);
        std.compute(x_std_gn, y, gn_cg);
        nform.compute(x_n_gn, y, gn_cg);
        mform.compute(x_m_gn, y, gn_cg);

        e1 = maximum_error(x_std_lm, x_std_gn);
        e2 = maximum_error(x_std_gn, x_n_gn);
        e3 = maximum_error(x_std_gn, x_m_gn);

        ensure_small(e1, "Error standard LM - standard GN");
        ensure_small(e2, "Error standard GN - n-form GN CG");
        ensure_small(e3, "Error standard GN - n-form GN CG ");

        // Test inversion using Preconditioned CG solver.
        using CGType = PreconditionedConjugateGradient<Preconditioner, false>;
        CGType pre_cg(1e-12);
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

        ensure_small(e1, "Error standard LM - standard GN");
        ensure_small(e2, "Error standard GN - n-form GN CGP");
        ensure_small(e3, "Error standard GN - n-form GN CGP ");
    }
};

TESTMAIN(GenericTest<LinearModel COMMA matrix_types>::run(10);)
