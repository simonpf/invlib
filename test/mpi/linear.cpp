#include "invlib/algebra.h"
#include "invlib/algebra/solvers.h"
#include "invlib/optimization.h"
#include "invlib/archetypes/vector_archetype.h"
#include "invlib/archetypes/matrix_archetype.h"
#include "invlib/mpi/mpi_vector.h"
#include "invlib/mpi/mpi_matrix.h"
#include "invlib/map.h"

#include "forward_models/linear.h"
#include "utility.h"

using namespace invlib;

int main()
{
    using RealType      = double;
    using VectorType    = Vector<VectorArchetype<double>>;
    using MatrixType    = Matrix<MatrixArchetype<double>>;
    using MPIMatrixType = Matrix<MPIMatrix<MatrixArchetype<double>, LValue>>;
    using Id     = MatrixIdentity<MatrixType>;
    using Model  = Linear<MPIMatrixType>;

    MPI_Init(nullptr, nullptr);

    unsigned int n = 50;

    MatrixType Se = random_positive_definite<MatrixType>(n);
    MatrixType Sa = random_positive_definite<MatrixType>(n);
    VectorType xa = random<VectorType>(n);
    VectorType y  = random<VectorType>(n);

    fill(y, 1.0);
    fill(xa, 1.0);

    fill(Sa, 0.0);
    fill(Se, 0.0);

    for (int i = 0; i < n; i++)
    {
        Sa(i, i) = 1.0;
        Se(i, i) = 1.0;
    }

    Model F(n,n);
    Model F_mpi(n,n);
    MAP<Model, MatrixType, MatrixType, MatrixType, Formulation::STANDARD>
        std(F, xa, Sa, Se);
    MAP<Model, MatrixType, MatrixType, MatrixType, Formulation::NFORM>
        nform(F, xa, Sa, Se);
    MAP<Model, MatrixType, MatrixType, MatrixType, Formulation::MFORM>
        mform(F, xa, Sa, Se);

    // Test inversion using CG solver.
    Id I{};
    ConjugateGradient cg(1e-9, 2);
    GaussNewton<RealType, ConjugateGradient> gn_cg(cg);
    gn_cg.set_tolerance(1e-9); gn_cg.set_maximum_iterations(1000);
    LevenbergMarquardt<RealType, Id, ConjugateGradient> lm_cg(I, cg);
    lm_cg.set_tolerance(1e-9); lm_cg.set_maximum_iterations(1000);

    VectorType x_std, x_m, x_n;


    std.compute(x_std, y, lm_cg, 2);
    mform.compute(x_m, y, gn_cg, 2);
    nform.compute(x_n, y, gn_cg, 2);

    auto e1 = maximum_error(x_std, x_m);
    auto e2 = maximum_error(x_std, x_n);

    std::cout << "Error STD - NFORM CG = " << e1 << std::endl;
    std::cout << "Error STD - MFORM CG = " << e2 << std::endl;

    MPI_Finalize();
}
