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

    using RealType      = float;
    using StdMatrix     = Matrix<MatrixArchetype<RealType>>;
    using StdVector     = Vector<VectorArchetype<RealType>>;
    using PllMatrix     = Matrix<MpiMatrix<MatrixArchetype<RealType>, LValue>>;
    using PllVector     = Vector<MpiVector<VectorArchetype<RealType>, LValue>>;
    using RefMatrix     = Matrix<MpiMatrix<MatrixArchetype<RealType>, ConstRef>>;
    using RefVector     = Vector<MpiVector<VectorArchetype<RealType>, ConstRef>>;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_m(5, 100);
    std::uniform_int_distribution<> dis_n(5, 100);

    typename StdMatrix::RealType max_err = 0.0;

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

        StdVector v; v.resize(n);
        fill(v, 1.0);

        StdVector w{}; w.resize(m);
        StdVector w_mpi{}; w.resize(m);

        PllMatrix M = random<PllMatrix>(m, n);
        RefMatrix MRef(M.get_local());

        w     = M * v;
        w_mpi = MRef * v;

        RealType err = maximum_error(w, w_mpi);
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

        StdVector v; v.resize(n);
        fill(v, 1.0);

        StdVector w{}; w.resize(n);
        StdVector w_mpi{}; w_mpi.resize(n);

        PllMatrix M = random<PllMatrix>(n, m);
        RefMatrix MRef(M.get_local());

        w     = transp(M) * v;
        w_mpi = transp(MRef) * v;

        RealType err = maximum_error(w, w_mpi);
        if (err > max_err)
            max_err = err;
    }

    if (rank == 0)
    {
        std::cout << "Testing MPI multiply: Max. rel. error = ";
        std::cout << max_err << std::endl;
    }

     // ------------------------- //
     //       Dot Prodcut         //
     // ------------------------- //

    for (int i = 0; i < n_tests; i++)
    {
        int m = dis_m(gen);
        MPI_Bcast(&m ,1, MPI_INTEGER, 0, MPI_COMM_WORLD);

        StdVector v       = random<StdVector>(m);
        RefVector v_mpi(v);

        int nprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        auto dot_1 = ((RealType) nprocs) * dot(v, v);
        auto dot_2 = dot(v_mpi, v_mpi);

        auto err = std::abs(dot_1 - dot_2) / std::max(std::abs(dot_1),
                                                      std::abs(dot_2));
        if (err > max_err)
            max_err = err;
    }

    if (rank == 0)
    {
        std::cout << "Testing MPI dot: Max. rel. error = ";
        std::cout << max_err << std::endl;
    }

    MPI_Finalize();

}
