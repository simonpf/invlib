// Copyright (C) 2016 Simon Pfreundschuh - All Rights Reserved
// For licensing term see LICENSE file in the root of this source tree.

//////////////////////////////////////////////////////////////
// Test MAP implementation for MPI-parallelized data types. //
//////////////////////////////////////////////////////////////

#include "invlib/algebra.h"
#include "invlib/algebra/solvers.h"
#include "invlib/optimization.h"
#include "invlib/algebra/precision_matrix.h"
#include "invlib/archetypes/vector_archetype.h"
#include "invlib/archetypes/matrix_archetype.h"
#include "invlib/mpi/log.h"
#include "invlib/mpi/mpi_vector.h"
#include "invlib/mpi/mpi_matrix.h"
#include "invlib/map.h"

#include "forward_models/linear.h"
#include "utility.h"

using namespace invlib;

/*
 * Generate random linear foward models with the MPI data types and
 * solve them.  If correct all computations should converge after one
 * step and yield the same result.
 */
int main()
{
    using RealType        = double;
    using VectorType      = Vector<VectorArchetype<RealType>>;
    using MatrixType      = Matrix<MatrixArchetype<RealType>>;
    using MpiVectorType   = Matrix<MpiVector<VectorArchetype<RealType>, LValue>>;
    using MpiMatrixType   = Matrix<MpiMatrix<MatrixArchetype<RealType>, LValue>>;
    using PrecisionMatrixType = PrecisionMatrix<MpiMatrixType>;
    using Id     = MatrixIdentity<MatrixType>;
    using Model  = Linear<MpiMatrixType>;

    MPI_Init(nullptr, nullptr);

    unsigned int n = 100;

    MatrixType SeInv_local; SeInv_local.resize(n,n);
    MatrixType SaInv_local; SaInv_local.resize(n,n);
    set_identity(SeInv_local);
    set_identity(SaInv_local);
    MpiMatrixType SeInv = MpiMatrixType::split_matrix(SeInv_local);
    MpiMatrixType SaInv = MpiMatrixType::split_matrix(SaInv_local);
    PrecisionMatrixType Pe(SeInv);
    PrecisionMatrixType Pa(SaInv);


    Model F(n,n);
    VectorType xa = random<MpiVectorType>(n);

    VectorType y = random<MpiVectorType>(n).gather();
    y = F.evaluate(y);

    MAP<Model, MatrixType, PrecisionMatrixType, PrecisionMatrixType,
        VectorType, Formulation::STANDARD>
        std(F, xa, Pa, Pe);
    MAP<Model, MatrixType, PrecisionMatrixType, PrecisionMatrixType,
        VectorType, Formulation::NFORM>
        nform(F, xa, Pa, Pe);

    // Test inversion using CG solver.
    using GN = GaussNewton<RealType, ConjugateGradient<>>;
    using LM = LevenbergMarquardt<RealType, Id, ConjugateGradient<>>;

    Id I{};
    ConjugateGradient<> cg(1e-9);
    GN gn_cg(1e-3, 2, cg);
    gn_cg.set_tolerance(1e-9); gn_cg.set_maximum_iterations(2);
    LM lm_cg(I, cg);
    lm_cg.set_tolerance(1e-9); lm_cg.set_maximum_iterations(1000);

    VectorType x_std, x_m, x_n;

    std.compute<GN,   MpiLog>(x_std, y, gn_cg, 2);
    nform.compute<GN, MpiLog>(x_n, y, gn_cg, 2);

    auto e1 = maximum_error(x_std, x_n);
    std::cout << "Error STD - NFORM CG = " << e1 << std::endl;

    MPI_Finalize();
}
