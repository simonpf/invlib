#include "invlib/algebra.h"
#include "utility.h"
#include "test_types.h"

using namespace invlib;

template <typename MatrixType>
struct Diagonal {

    static constexpr char name[] = "Diagonal";

    static void run(size_t m) {
        using VectorType = typename MatrixType::VectorType;
        MatrixType A = random_diagonal<MatrixType>(m);
        MatrixType B = random_diagonal<MatrixType>(m);
        MatrixType C = random_diagonal<MatrixType>(m);

        VectorType v, w;

        C = A * B;
        v = C.diagonal();
        w = (A * B).diagonal();
        double error = maximum_error(v, w);
        ensure_small(error, "Diagonal of product");

        C = A + B;
        v = C.diagonal();
        w = (A + B).diagonal();
        error = maximum_error(v, w);
        ensure_small(error, "Diagonal of sum");

        C = transp(A) * B * A + 3.0 * A;
        v = C.diagonal();
        w = (transp(A) * B * A + 3.0 * A).diagonal();
        error = maximum_error(v, w);
        ensure_small(error, "OEM diagonal");

        C = (A * ((A * B) + (B * B) - A));
        v = C.diagonal();
        w = (A * ((A * B) + (B * B) - A)).diagonal();
        error = maximum_error(v, w);
        ensure_small(error, "Complex expression diagonal");
    }
};

TESTMAIN(GenericTest<Diagonal COMMA matrix_types>::run(100);)
