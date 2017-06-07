#include <type_traits>

#include "invlib/interfaces/eigen.h"
#include "invlib/algebra.h"
#include "invlib/io.h"
#include "invlib/map.h"
#include "invlib/optimization/gauss_newton.h"
#include "invlib/traits.h"


using MatrixType = invlib::Matrix<invlib::EigenSparse>;
using VectorType = invlib::Vector<invlib::EigenVector>;

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

template<typename T> void foo(T);

int main()
{

    using SolverType      = invlib::ConjugateGradient<>;
    using MinimizerType   = invlib::GaussNewton<double, SolverType>;
    using PrecisionMatrix = invlib::PrecisionMatrix<MatrixType>;
    using MAPType         = invlib::MAP<LinearModel,
                                        MatrixType,
                                        PrecisionMatrix,
                                        PrecisionMatrix>;

    // Load data.
    MatrixType K     = invlib::read_matrix_arts("STR_VALUE(MATS_DATA)/K.xml");
    MatrixType SaInv = invlib::read_matrix_arts("STR_VALUE(MATS_DATA)/SaInv.xml");
    MatrixType SeInv = invlib::read_matrix_arts("STR_VALUE(MATS_DATA)/SeInv.xml");
    PrecisionMatrix Pa(SaInv);
    PrecisionMatrix Pe(SeInv);

    VectorType y     = invlib::read_vector_arts("STR_VALUE(MATS_DATA)/y.vec");
    VectorType xa    = invlib::read_vector_arts("STR_VALUE(MATS_DATA)/xa.vec");

    // Setup OEM.
    SolverType    cg(1e-6, 1);
    MinimizerType gn(1e-6, 1, cg);
    LinearModel   F(K, xa);
    MAPType       oem(F, xa, Pa, Pe);

    // Run OEM.
    VectorType x;
    oem.compute(x, y, gn, 0);

    invlib::write_vector_arts("x.xml", (invlib::VectorData<double>) x, invlib::Format::ASCII);

    return 0.0;
}

