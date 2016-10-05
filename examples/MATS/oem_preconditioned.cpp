#include "invlib/map.h"
#include "eigen.h"
#include "invlib/algebra/precision_matrix.h"
#include "invlib/algebra/solvers.h"
#include "invlib/optimization/gauss_newton.h"

#include "eigen_io.h"

using MatrixType = invlib::Matrix<EigenSparse>;
using VectorType = invlib::Vector<EigenVector>;

class LinearModel
{
public:

    LinearModel(const MatrixType &K_, const VectorType &xa_)
        : K(K_), xa(xa_), m(K_.rows()), n(K_.cols())
    {
        // Nothing to do here.
    }

    VectorType evaluate(const VectorType &x)
    {
        return K * (x - xa);
    }

    const MatrixType & Jacobian(const VectorType &x, VectorType &y)
    {
        y = K * (x - xa);
        return K;
    }

    const unsigned int m, n;

private:

    const MatrixType &K;
    const VectorType &xa;

};

// Implementation of a Jacobian preconditioner for the Eigen3 backend
// which is necessary due to performance reasons.
struct JacobianPreconditioner
{
    JacobianPreconditioner(const EigenSparse & K,
                           const EigenSparse & SaInv,
                           const EigenSparse & SeInv)
    {
        diag = SaInv.diagonal();
        for (size_t i = 0; i < K.cols(); i++)
        {
            std::cout << i << std::endl;
            diag(i) += K.col(i).dot(SeInv.diagonal() * K.col(i));
        }
        diag.cwiseInverse();
    }

    VectorType operator()(const VectorType &v) const
    {
        VectorType w; w.resize(v.rows());
        w = diag.array() * v.array();
        return w;
    }

private:

    VectorType diag;

};

int main()
{

    using MatrixType = invlib::Matrix<EigenSparse>;
    using VectorType = invlib::Vector<EigenVector>;

    // Un-cached (seconde template argument = false), preconditioned CG solver.
    // Un-cached meaning that the preconditioner is reinitialized, i.e. the
    // diagonal of the Jacobian preconditioner recomputed, on each invocation.
    using SolverType = invlib::PreconditionedConjugateGradient<JacobianPreconditioner,
                                                               true>;
    using MinimizerType   = invlib::GaussNewton<double, SolverType>;
    using PrecisionMatrix = invlib::PrecisionMatrix<MatrixType>;
    using MAPType         = invlib::MAP<LinearModel,
                                        MatrixType,
                                        PrecisionMatrix,
                                        PrecisionMatrix>;

    // Load data.
    MatrixType K     = read_sparse_matrix("data/K.sparse");
    MatrixType SaInv = read_sparse_matrix("data/SaInv.sparse");
    MatrixType SeInv = read_sparse_matrix("data/SeInv.sparse");
    PrecisionMatrix Pa(SaInv);
    PrecisionMatrix Pe(SeInv);

    VectorType y     = read_vector("data/y.vec");
    VectorType xa    = read_vector("data/xa.vec");

    // Setup the preconditioner.
    JacobianPreconditioner pre(K, SaInv, SeInv);

    // Setup OEM.
    SolverType    cg(pre, 1e-6, 1);
    MinimizerType gn(1e-6, 1, cg);
    LinearModel   F(K, xa);
    MAPType       oem(F, xa, Pa, Pe);

    // Run OEM.
    VectorType x;
    oem.compute(x, y, gn, 0);

    write_vector(x, "x.vec");
}

