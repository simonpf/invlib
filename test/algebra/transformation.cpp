#include "invlib/algebra.h"
#include "utility.h"
#include "test_types.h"

using namespace invlib;

template <typename MatrixType>
struct Transformations {

    using RealType   = typename MatrixType::RealType;
    using VectorType = Vector<typename MatrixType::VectorType>;

    static constexpr char name[] = "Transformations";

    static void run(size_t n) {

        auto A = random_positive_definite<MatrixType>(n);
        auto B = random_positive_definite<MatrixType>(n);
        auto v = random<VectorType>(n);

        RealType error;

        // Identity Transformation
        Identity I{};
        VectorType w1 = I.apply(A * B) * I.apply(v);
        VectorType w2 = A * B * v;
        error = maximum_error(w1, w2);
        ensure_small(error, "Identity transformation.");

        // NormalizeDiagonal Transform
        NormalizeDiagonal<MatrixType> t(A);
        w1 = t.apply(inv(t.apply(A)) * t.apply(v));
        w2 = inv(A) * v;
        error = maximum_error(w1, w2);
        ensure_small(error, "Normalize diagonal");
    }
};

TESTMAIN(GenericTest<Transformations COMMA matrix_types>::run(100);)
