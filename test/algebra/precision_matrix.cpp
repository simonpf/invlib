#include "invlib/algebra.h"
#include "utility.h"
#include "test_types.h"

using namespace invlib;

// Test behaviour of precision matrix. The PrecisionMatrix wrapper should
// make a matrix act like its inverse. This is tested below by comparing
// the precision matrix constructed from A to its inverse and vice versa.
template <typename MatrixType>
struct InverseCovariance {

    static constexpr char name[] = "PrecisionMatrix";

    using VectorType = typename MatrixType::VectorType;

    static void run(size_t n) {
        MatrixType A  = random_positive_definite<MatrixType>(n);
        PrecisionMatrix<MatrixType> P(A);
        VectorType v = random<VectorType>(n);

        MatrixType B = P * A;
        MatrixType C = inv(P) * inv(A);
        MatrixType I; I.resize(n, n); set_identity(I);
        auto error = maximum_error(B, I);
        ensure_small(error, "Deviation from identity");
        error = maximum_error(C, I);
        ensure_small(error, "Deviation from identity");

        MatrixType D = P;
        MatrixType E = inv(A);
        error = maximum_error(D, E);
        ensure_small(error, "Deviation from from inv(A)");

        VectorType w1 = inv(A) * v;
        VectorType w2 = P * v;
        error = maximum_error(w1, w2);
        ensure_small(error, "Vector multiplication error");
    }
};

TESTMAIN(GenericTest<InverseCovariance COMMA matrix_types>::run(100);)
