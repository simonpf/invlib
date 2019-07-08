#include "invlib/algebra.h"
#include "invlib/algebra/solvers.h"
#include "invlib/algebra/preconditioners.h"
#include "utility.h"
#include "test_types.h"

using namespace invlib;

// Test solvers by computing A * inv(A) * v for a random vector v and a
// random positive definite matrix A. The resulting vector should be equal
// to v up to the precision of the underlying solver.
template <typename MatrixType>
struct Solver {

    static constexpr char name[] = "Conjugate gradient";

    static void run(unsigned int n) {

        using VectorType = Vector<typename MatrixType::VectorType>;
        auto A  = random_positive_definite<MatrixType>(n);
        auto v = random<VectorType>(n);
        VectorType w; w.resize(n);

        Standard std{};
        ConjugateGradient<> cg(1e-20);

        w = A * std.solve(A, v);
        auto error = maximum_error(v, w);
        ensure_small(error, "Standard inverse");

        w = A * cg.solve(A, v);
        error = maximum_error(v, w);
        ensure_small(error, "CG inverse");
    }
};

template <typename MatrixType>
struct PreconditionedSolver {

    static constexpr char name[] = "Conjugate gradient";

    static void run(unsigned int n) {

        using VectorType = Vector<typename MatrixType::VectorType>;
        using Preconditioner = JacobianPreconditioner<VectorType>;

        auto A  = random_positive_definite<MatrixType>(n);
        auto v = random<VectorType>(n);
        VectorType w; w.resize(n);

        Standard std{};
        JacobianPreconditioner<VectorType> pre(A);
        PreconditionedConjugateGradient<Preconditioner, true> cg(pre, 1e-20);

        w = A * std.solve(A, v);
        auto error = maximum_error(v, w);
        ensure_small(error, "Standard inverse");

        w = A * cg.solve(A, v);
        error = maximum_error(v, w);
        ensure_small(error, "CG inverse");
    }
};

TESTMAIN(GenericTest<Solver COMMA matrix_types>::run(100);
         GenericTest<PreconditionedSolver COMMA matrix_types>::run(100);)
