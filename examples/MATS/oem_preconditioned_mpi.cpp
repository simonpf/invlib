#include "invlib/map.h"
#include "eigen_mpi.h"
#include "invlib/algebra/precision_matrix.h"
#include "invlib/algebra/solvers.h"
#include "invlib/optimization/gauss_newton.h"
#include "invlib/mpi.h"

#include "eigen_io.h"

using MatrixType = invlib::Matrix<EigenSparse>;
using VectorType = invlib::Vector<EigenVector>;
using MPIMatrixType = invlib::Matrix<invlib::MPIMatrix<EigenSparse, invlib::LValue>>;
using MPIVectorType = invlib::Vector<invlib::MPIVector<EigenVector, invlib::LValue>>;

class LinearModel
{
public:

    LinearModel(const MPIMatrixType &K_, const MPIVectorType &xa_)
        : K(K_), xa(xa_), m(K_.rows()), n(K_.cols())
    {
        // Nothing to do here.
    }

    MPIVectorType evaluate(const MPIVectorType &x)
    {
        return K * (x - xa);
    }

    const MPIMatrixType & Jacobian(const MPIVectorType &x, MPIVectorType &y)
    {
        y = K * (x - xa);
        return K;
    }

    unsigned int m, n;

private:

    const MPIMatrixType &K;
    const MPIVectorType &xa;

};

// Implementation of a Jacobian preconditioner for the Eigen3 backend
// which is necessary due to performance reasons.
struct JacobianPreconditioner
{
    JacobianPreconditioner(const MatrixType & K,
                           const MatrixType & SaInv,
                           const MatrixType & SeInv)
        : diag()
    {
        diag.resize(SaInv.rows());
        VectorType diag_full = SaInv.diagonal();
        diag = diag_full.get_block(diag.get_index(), diag.get_range());
        for (size_t i = diag.get_index(); i < diag.get_range(); i++)
        {
            size_t j = i - diag.get_index();
            diag.get_local().operator()(j) +=
                K.col(i).dot(SeInv.diagonal() * K.col(i));
        }
        diag.get_local().cwiseInverse();
    }

    MPIVectorType operator()(const MPIVectorType &v) const
    {
        MPIVectorType w; w.resize(v.rows());
        w.get_local() = diag.get_local().array() * v.get_local().array();
        return w;
    }

private:

    MPIVectorType diag;

};

int main()
{

    using SolverType = invlib::PreconditionedConjugateGradient<JacobianPreconditioner,
                                                               true>;
    using MinimizerType   = invlib::GaussNewton<double, SolverType>;
    using PrecisionMatrix = invlib::PrecisionMatrix<MPIMatrixType>;
    using MAPType         = invlib::MAP<LinearModel,
                                        MPIMatrixType,
                                        PrecisionMatrix,
                                        PrecisionMatrix,
                                        MPIVectorType>;

    // Initialize MPI.
    MPI_Init(nullptr, nullptr);

    // Load data.
    MatrixType K     = read_sparse_matrix("data/K.sparse");
    MatrixType SaInv = read_sparse_matrix("data/SaInv.sparse");
    MatrixType SeInv = read_sparse_matrix("data/SeInv.sparse");

    MPIMatrixType K_mpi     = MPIMatrixType::split_matrix(K);
    MPIMatrixType SaInv_mpi = MPIMatrixType::split_matrix(SaInv);
    MPIMatrixType SeInv_mpi = MPIMatrixType::split_matrix(SeInv);

    PrecisionMatrix Pa(SaInv_mpi);
    PrecisionMatrix Pe(SeInv_mpi);

    VectorType y     = read_vector("data/y.vec");
    VectorType xa    = read_vector("data/xa.vec");

    MPIVectorType y_mpi  = MPIVectorType::split(y);
    MPIVectorType xa_mpi = MPIVectorType::split(xa);

    // Setup preconditioner.
    JacobianPreconditioner pre(K, SaInv, SeInv);

    // Setup OEM.
    SolverType    cg(pre, 1e-6, 1);
    MinimizerType gn(1e-6, 1, cg);
    LinearModel   F(K_mpi, xa_mpi);
    MAPType       oem(F, xa_mpi, Pa, Pe);

    // Run OEM.
    MPIVectorType x_mpi{};
    oem.compute<MinimizerType, invlib::MPILog>(x_mpi, y_mpi, gn, 0);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        write_vector(x_mpi, "x.vec");

    MPI_Finalize();
}
