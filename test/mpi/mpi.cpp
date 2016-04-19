#include <random>

#include "invlib/mpi/mpi_matrix.h"
#include "invlib/archetypes/matrix_archetype.h"
#include "utility.h"

using namespace invlib;

int main()
{

    // MPI Setup.

    MPI_Init(nullptr, nullptr);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    using Matrix  = Matrix<MatrixArchetype<double>>;
    using Vector  = Vector<VectorArchetype<double>>;
    using DMatrix = MPIMatrix<Matrix, LValue>;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_m(nprocs, 10);
    std::uniform_int_distribution<> dis_n(1, 10);

    double max_err = 0.0;

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

        Vector v; v.resize(n);
        fill(v, 1.0);

        Vector w{}; w.resize(m);
        Vector w_mpi{}; w.resize(m);

        auto M = random<Matrix>(m, n);
        DMatrix::broadcast(M);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        DMatrix DM = DMatrix::split_matrix(M);

        w     = M.multiply(v);
        w_mpi = DM.multiply(v);


        double err = maximum_error(w, w_mpi);
        if (err > max_err)
            max_err = err;
    }

    if (rank == 0)
    {
        std::cout << "Testing MPI multiply: Max. rel. error = ";
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

        Vector v; v.resize(m);
        fill(v, 1.0);

        Vector w{}; w.resize(n);
        Vector w_mpi{}; w.resize(n);

        auto M = random<Matrix>(m, n);
        DMatrix::broadcast(M);

        DMatrix DM = DMatrix::split_matrix(M);

        w     = M.transpose_multiply(v);
        w_mpi = DM.transpose_multiply(v);

        double err = maximum_error(w, w_mpi);
        if (err > max_err)
            max_err = err;
    }

    if (rank == 0)
    {
        std::cout << "Testing MPI transpose_multiply: Max. rel. error = ";
        std::cout << max_err << std::endl;
    }
    MPI_Finalize();

}
