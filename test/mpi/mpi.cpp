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
    std::uniform_int_distribution<> dis_m(2, 500);
    std::uniform_int_distribution<> dis_n(2, 500);

    typename DMatrix::RealType max_err = 0.0;

    // -------------------- //
    // Standard Matrix Mult //
    // -------------------- //

    int n_tests = 100;
    for (int i = 0; i < n_tests; i++)
    {
        int m = dis_m(gen);
        int n = dis_n(gen);

        MPI_Bcast(&m ,1, MPI_INTEGER, 0, MPI_COMM_WORLD);
        MPI_Bcast(&n ,1, MPI_INTEGER, 0, MPI_COMM_WORLD);

        DVector v; v.resize(n);
        SVector v_mpi{}; v_mpi.resize(n);
        fill(v, 1.0);
        fill(v_mpi, 1.0);

        DVector w;
        DVector w_mpi;
        SVector w_v_mpi;

        auto M = random<DMatrix>(m, n);
        SMatrix::broadcast(M);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        SMatrix SM = SMatrix::split_matrix(M);

        w       = M * v;
        w_mpi   = SM * v;
        w_v_mpi = SM * v_mpi;


        double err = maximum_error(w, w_mpi);
        DVector temp = w_v_mpi.broadcast();

        err = std::max(err, maximum_error(w, temp));
        if (err > max_err)
            max_err = err;
    }

    if (rank == 0)
    {
        std::cout << "Testing MPI multiply:           Max. rel. error = ";
        std::cout << max_err << std::endl;
    }

    // ------------------------- //
    // Transposed Multiplication //
    // ------------------------- //

    for (int i = 0; i < n_tests; i++)
    {
        int m = dis_m(gen);
        int n = dis_n(gen);

        MPI_Bcast(&m ,1, MPI_INTEGER, 0, MPI_COMM_WORLD);
        MPI_Bcast(&n ,1, MPI_INTEGER, 0, MPI_COMM_WORLD);

        DVector v; v.resize(m);
        fill(v, 1.0);

        DVector w{};
        DVector w_mpi{};

        auto M = random<DMatrix>(m, n);
        SMatrix::broadcast(M);

        SMatrix SM = SMatrix::split_matrix(M);

        w     = transp(M) * v;
        w_mpi = transp(SM) * v;

        double err = maximum_error(w, w_mpi);
        if (err > max_err)
            max_err = err;
    }

    if (rank == 0)
    {
        std::cout << "Testing MPI transpose_multiply: Max. rel. error = ";
        std::cout << max_err << std::endl;
    }

    // ------------------------- //
    //       Dot Prodcut         //
    // ------------------------- //

    max_err = 0.0;

    for (int i = 0; i < n_tests; i++)
    {
        int m = dis_m(gen);
        MPI_Bcast(&m ,1, MPI_INTEGER, 0, MPI_COMM_WORLD);
        SVector v       = random<SVector>(m);
        DVector v_local = v.broadcast();

        auto dot_1 = dot(v, v);
        auto dot_2 = dot(v_local, v_local);

        auto err = std::abs(dot_1 - dot_2) / std::max(std::abs(dot_1),
                                                      std::abs(dot_2));
        if (err > max_err)
            max_err = err;
    }

    if (rank == 0)
    {
        std::cout << "Testing MPI dot:                Max. rel. error = ";
        std::cout << max_err << std::endl;
    }

    MPI_Finalize();
}
