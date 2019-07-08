#include "invlib/algebra.h"
#include "utility.h"
#include "test_types.h"

using namespace invlib;

template <typename MatrixType>
struct Identities {

    static constexpr char name[] = "Identities";

    static void run(size_t k, size_t m, size_t n) {
        auto A = random<MatrixType>(m, k);
        auto B = random<MatrixType>(m, k);
        auto C = random<MatrixType>(k, n);
        auto D = random<MatrixType>(k, n);

        // Addition.

        MatrixType R1 = A + B;
        MatrixType R2 = B + A;
        double error = maximum_error(R1, R2);
        ensure_small(error, "Commutativity of matrix sum");

        R1 = (A + B) + B;
        R2 = B + (A + B);
        error = maximum_error(R1, R2);
        ensure_small(error, "Associativity of matrix sum");

        // Subtraction.

        R1 = A - B;
        R2 = -1.0 * (B - A);
        ensure_small(error, "Subtraction of matrices");

            R1 = (B - A) + B;
        R2 = B - (A - B);

        error = maximum_error(R1, R2);
        ensure_small(error, "Associativity of sum/difference");

        // Multiplication.

        R1 = transp(A * C);
        R2 = transp(C) * transp(A);
        error = maximum_error(R1, R2);
        ensure_small(error, "Transpose of product");

        R1 = (A + B) * (C + D);
        R2 = A*C + A*D + B*C + B*D;
        error = maximum_error(R1, R2);
        ensure_small(error, "Distributive property sum");

        R1 = (A - B) * (C - D);
        R2 = A*C - A*D - B*C + B*D;
        error = maximum_error(R1, R2);
        ensure_small(error, "Distributive property difference");

        R1 = transp((A + B) * (C + D));
        R2 = transp(A*C) + transp(A*D) + transp(B*C) + transp(B*D);
        error = maximum_error(R1, R2);
        ensure_small(error, "Distributive property of transposed sum");

        R2 = transp(C)*transp(A) + transp(D)*transp(A)
            + transp(C)*transp(B) + transp(D)*transp(B);
        error = maximum_error(R1, R2);
        ensure_small(error, "Distributive property of transposed sum");

            // Inversion.

        MatrixType E = random_positive_definite<MatrixType>(m);
        R1 = inv(E) * E;
        R2 = E * inv(E);
        set_identity(E);
        error = maximum_error(R1, E);
        ensure_small(error, "Left matrix inverse");
        error = maximum_error(R2, E);
        ensure_small(error, "Right matrix inverse");
    }
};

TESTMAIN(GenericTest<Identities COMMA matrix_types>::run(100, 50, 20);)
