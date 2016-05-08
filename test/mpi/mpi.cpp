#include <random>

#include "invlib/mpi/mpi_matrix.h"
#include "invlib/mpi/mpi_vector.h"
#include "invlib/archetypes/matrix_archetype.h"
#include "invlib/algebra/solvers.h"
#include "utility.h"

using namespace invlib;

int main()
{

    // MPI Setup.
    MPI_Init(nullptr, nullptr);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // D for duplicated types.
    // S for distributed (split) types.
    using DMatrix  = Matrix<MatrixArchetype<double>>;
    using DVector  = Vector<VectorArchetype<double>>;
    using SMatrix  = Matrix<MPIMatrix<MatrixArchetype<double>, LValue>>;
    using SVector  = Vector<MPIVector<VectorArchetype<double>, LValue>>;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_m(1, 500);
    std::uniform_int_distribution<> dis_n(1, 500);

    typename DMatrix::RealType max_err = 0.0;
    DMatrix M = random<DMatrix>(4,8);
    SMatrix::broadcast(M);
    SMatrix MM = SMatrix::split_matrix(M);

    SVector v; v.resize(4);

    for (int i = 0; i < 4; i++)
        v(i) = 0.0;
    v(2) = 1.0;

    SVector w  = transp(MM) * v;

    DVector w2  = w;

    if (rank == 0)
    {
        std::cout << M << std::endl;
        std::cout << w.rows() << std::endl;
        std::cout << w2 << std::endl;
    }

    // -------------------- //
    // Standard Matrix Mult //
    // -------------------- //

    // int n_tests = 100;
    // for (int i = 0; i < n_tests; i++)
    // {
    //     int m = dis_m(gen);
    //     int n = dis_n(gen);

    //     MPI_Bcast(&m ,1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    //     MPI_Bcast(&n ,1, MPI_INTEGER, 0, MPI_COMM_WORLD);

    //     DVector v; v.resize(n);
    //     fill(v, 1.0);

    //     DVector w{}; w.resize(m);
    //     DVector w_mpi{}; w.resize(m);

    //     auto M = random<DMatrix>(m, n);
    //     SMatrix::broadcast(M);

    //     MPI_Barrier(MPI_COMM_WORLD);
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     SMatrix SM = SMatrix::split_matrix(M);

    //     w     = M * v;
    //     w_mpi = SM * v;


    //     double err = maximum_error(w, w_mpi);
    //     if (err > max_err)
    //         max_err = err;
    // }

    // if (rank == 0)
    // {
    //     std::cout << "Testing MPI multiply: Max. rel. error = ";
    //     std::cout << max_err << std::endl;
    // }

    // // ------------------------- //
    // // Transposed Multiplication //
    // // ------------------------- //

    // for (int i = 0; i < n_tests; i++)
    // {
    //     int m = dis_m(gen);
    //     int n = dis_n(gen);

    //     MPI_Bcast(&m ,1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    //     MPI_Bcast(&n ,1, MPI_INTEGER, 0, MPI_COMM_WORLD);

    //     DVector v; v.resize(m);
    //     fill(v, 1.0);

    //     DVector w{}; w.resize(n);
    //     DVector w_mpi{}; w.resize(n);

    //     auto M = random<DMatrix>(m, n);
    //     SMatrix::broadcast(M);

    //     SMatrix SM = SMatrix::split_matrix(M);

    //     w     = transp(M) * v;
    //     w_mpi = transp(SM) * v;

    //     double err = maximum_error(w, w_mpi);
    //     if (err > max_err)
    //         max_err = err;
    // }

    // if (rank == 0)
    // {
    //     std::cout << "Testing MPI transpose_multiply: Max. rel. error = ";
    //     std::cout << max_err << std::endl;
    // }

    MPI_Finalize();

}
