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
    std::uniform_int_distribution<> dis_m(nprocs, 100);
    std::uniform_int_distribution<> dis_n(nprocs, 100);

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

        DVector v;  v.resize(m);
        SVector vv; vv.resize(m);
        fill(v,  1.0);
        fill(vv, 1.0);

        DVector w, w_{};
        SVector ww{};
        DVector w_mpi{};

        auto M = random<DMatrix>(m, n);
        SMatrix::broadcast(M);

        SMatrix SM = SMatrix::split_matrix(M);

        w  = transp(M)  * v;
        w_ = transp(SM) * v;
        ww = transp(SM) * vv;

        double err = maximum_error(w, w_);
        if (err > max_err)
            max_err = err;

        DVector ww_ = ww.broadcast();
        err = maximum_error(w, ww_);
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
        //auto dot_3 = dot(v, v_local);
        //auto dot_4 = dot(v_local, v);

        auto err = std::abs(dot_1 - dot_2) / std::max(std::abs(dot_1),
                                                      std::abs(dot_2));
        if (err > max_err)
            max_err = err;

        //err = std::abs(dot_1 - dot_3) / std::max(std::abs(dot_1), std::abs(dot_3));
        //if (err > max_err)
        //    max_err = err;

        //err = std::abs(dot_1 - dot_4) / std::max(std::abs(dot_1), std::abs(dot_4));
        //if (err > max_err)
        //    max_err = err;
    }

    if (rank == 0)
    {
        std::cout << "Testing MPI dot:                Max. rel. error = ";
        std::cout << max_err << std::endl;
    }

    // ------------------------- //
    //       Dot Prodcut         //
    // ------------------------- //

    max_err = 0.0;

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

        auto M_1 = random<DMatrix>(m, n);
        SMatrix::broadcast(M_1);
        auto M_2 = random<DMatrix>(m, m);
        SMatrix::broadcast(M_2);

        SMatrix SM_1 = SMatrix::split_matrix(M_1);
        SMatrix SM_2 = SMatrix::split_matrix(M_2);

        DMatrix M_4 = transp(M_1) * M_2 * M_1;
        DVector diag_1 = M_4.diagonal();
        DVector diag_2 = (transp(SM_1) * SM_2 * SM_1).diagonal();

        auto err = maximum_error<DVector>(diag_1, diag_2);
        if (err > max_err)
            max_err = err;
    }

    if (rank == 0)
    {
        std::cout << "Testing MPI diagonal:           Max. rel. error = ";
        std::cout << max_err << std::endl;
    }

    MPI_Finalize();
}
