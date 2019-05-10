#include "invlib/map.h"
#include "invlib/interfaces/eigen.h"
#include "invlib/algebra/precision_matrix.h"
#include "invlib/algebra/solvers.h"
#include "invlib/optimization/gauss_newton.h"
#include "invlib/mpi.h"

#include "eigen_io.h"

using invlib::EigenVector;
using invlib::EigenSparse;
using MatrixType = invlib::Matrix<EigenSparse>;
using VectorType = invlib::Vector<EigenVector>;
using MpiMatrixType = invlib::Matrix<invlib::MpiMatrix<EigenSparse, invlib::LValue>>;
using MpiVectorType = invlib::Vector<invlib::MpiVector<EigenVector, invlib::LValue>>;

class LinearModel
{
public:

    LinearModel(const MpiMatrixType &K_, const MpiVectorType &xa_)
        : K(K_), xa(xa_), m(K_.rows()), n(K_.cols())
    {
        // Nothing to do here.
    }

    MpiVectorType evaluate(const MpiVectorType &x)
    {
        return K * (x - xa);
    }

    const MpiMatrixType & Jacobian(const MpiVectorType &x, MpiVectorType &y)
    {
        y = K * (x - xa);
        return K;
    }

    unsigned int m, n;

private:

    const MpiMatrixType &K;
    const MpiVectorType &xa;

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

    MpiVectorType operator()(const MpiVectorType &v) const
    {
        MpiVectorType w; w.resize(v.rows());
        w.get_local() = diag.get_local().array() * v.get_local().array();
        return w;
    }

private:

    MpiVectorType diag;

};

int main()
{

    using SolverType = invlib::PreconditionedConjugateGradient<JacobianPreconditioner,
                                                               true>;
    using MinimizerType   = invlib::GaussNewton<double, SolverType>;
    using PrecisionMatrix = invlib::PrecisionMatrix<MpiMatrixType>;
    using MAPType         = invlib::MAP<LinearModel,
                                        MpiMatrixType,
                                        PrecisionMatrix,
                                        PrecisionMatrix,
                                        MpiVectorType>;

    // Initialize MPI.
    MPI_Init(nullptr, nullptr);

    // Load data.
    MatrixType K     = read_sparse_matrix("data/K.sparse");
    MatrixType SaInv = read_sparse_matrix("data/SaInv.sparse");
    MatrixType SeInv = read_sparse_matrix("data/SeInv.sparse");

    MpiMatrixType K_mpi     = MpiMatrixType::split_matrix(K);
    MpiMatrixType SaInv_mpi = MpiMatrixType::split_matrix(SaInv);
    MpiMatrixType SeInv_mpi = MpiMatrixType::split_matrix(SeInv);

    PrecisionMatrix Pa(SaInv_mpi);
    PrecisionMatrix Pe(SeInv_mpi);

    VectorType y     = read_vector("data/y.vec");
    VectorType xa    = read_vector("data/xa.vec");

    MpiVectorType y_mpi  = MpiVectorType::split(y);
    MpiVectorType xa_mpi = MpiVectorType::split(xa);

    // Setup preconditioner.
    JacobianPreconditioner pre(K, SaInv, SeInv);

    // Setup OEM.
    SolverType    cg(pre, 1e-6, 1);
    MinimizerType gn(1e-6, 1, cg);
    LinearModel   F(K_mpi, xa_mpi);
    MAPType       oem(F, xa_mpi, Pa, Pe);

    // Run OEM.
    MpiVectorType x_mpi{};
    oem.compute<MinimizerType, invlib::MpiLog>(x_mpi, y_mpi, gn, 0);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        write_vector(x_mpi, "x.vec");

    MPI_Finalize();

    return 0.0;
}
