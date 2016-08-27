#include <iostream>

#include "invlib/algebra.h"
#include "invlib/algebra/solvers.h"
#include "invlib/archetypes/matrix_archetype.h"
#include "invlib/archetypes/vector_archetype.h"
#include "invlib/mpi.h"

#include "utility.h"

using namespace invlib;

template
<
typename T
>
void mpi_conjugate_gradient(unsigned int ntests)
{
    using RealType   = typename T::RealType;
    using VectorType = Vector<typename T::VectorType>;
    using MatrixType = Matrix<typename T::MatrixType>;
    using MPIVectorType = Vector<MPIVector<typename T::VectorType, LValue>>;
    using MPIMatrixType = Matrix<MPIMatrix<typename T::MatrixType, LValue>>;

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    RealType err, max_err = 0.0;

    for (unsigned int i = 0; i < ntests; i++)
    {

        MatrixType M        = random_positive_definite<MatrixType>(200);
        MPIMatrixType::broadcast(M);
        MPIMatrixType M_mpi = MPIMatrixType::split_matrix(M);

        MPIVectorType v_mpi = random<MPIVectorType>(200);
        VectorType v        = v_mpi;

        ConjugateGradient cg(1e-12, 0);

        VectorType     w = cg.solve<VectorType, MatrixType, MPILog>(M, v);
        VectorType w_mpi = cg.solve<MPIVectorType, MPIMatrixType, MPILog>(M_mpi,
                                                                             v_mpi);

        err = maximum_error(w_mpi, w);
        if (err > max_err)
            max_err = err;
    }

    if (rank == 0)
    {
        std::cout << "Testing MPI conjugate gradient: Max. rel. error = ";
        std::cout << max_err << std::endl;
    }
}

int main()
{
    // MPI Setup.
    MPI_Init(nullptr, nullptr);

    using MatrixType = MatrixArchetype<double>;
    mpi_conjugate_gradient<MatrixType>(10);

    MPI_Finalize();
}
