#include "invlib/map.h"
#include "eigen.h"
#include "invlib/algebra/precision_matrix.h"
#include "invlib/algebra/solvers.h"
#include "invlib/algebra/preconditioners.h"
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

int main()
{

    using MatrixType = invlib::Matrix<EigenSparse>;
    using VectorType = invlib::Vector<EigenVector>;

    // Jacobian preconditioner using VectorType to store the element-wise inverse
    // of the diagonal.
    using Preconditioner  = invlib::JacobianPreconditioner<VectorType>;
    // Un-cached (seconde template argument = false), preconditioned CG solver.
    // Un-cached meaning that the preconditioner is reinitialized, i.e. the
    // diagonal of the Jacobian preconditioner recomputed, on each invocation.
    using SolverType      = invlib::PreconditionedConjugateGradient<Preconditioner,
                                                                    false>;
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

    // Setup OEM.
    SolverType    cg(1e-6, 1);
    MinimizerType gn(1e-6, 1, cg);
    LinearModel   F(K, xa);
    MAPType       oem(F, xa, Pa, Pe);

    // Run OEM.
    VectorType x;
    oem.compute(x, y, gn, 0);

    write_vector(x, "x.vec");
}

