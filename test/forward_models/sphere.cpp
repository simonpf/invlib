#include <iostream>

#include "invlib/algebra.h"
#include "invlib/map.h"
#include "invlib/optimization.h"

#include "forward_models/sphere.h"
#include "utility.h"
#include "test_types.h"

using namespace invlib;

// Use the sphere function forward model to test the equivalence of the
// standard, n-form and m-form when using the Gauss-Newton optimizer.
template <typename T>
struct SphereModel {

    static constexpr char name[] = "Sphere model";

    static void run(size_t n) {
        using RealType   = typename T::RealType;
        using VectorType = Vector<typename T::VectorType>;
        using MatrixType = T;
        using Model      = Sphere<MatrixType>;

        MatrixType Se = random_positive_definite<MatrixType>(1);
        MatrixType Sa = random_positive_definite<MatrixType>(n);
        VectorType xa = random<VectorType>(n);
        VectorType y  = random<VectorType>(1);

        Model F(n);
        MAP<Model, MatrixType, MatrixType, MatrixType, VectorType, Formulation::STANDARD>
            std(F, xa, Sa, Se);
        MAP<Model, MatrixType, MatrixType, MatrixType, VectorType, Formulation::NFORM>
            nform(F, xa, Sa, Se);
        MAP<Model, MatrixType, MatrixType, MatrixType, VectorType, Formulation::MFORM>
            mform(F, xa, Sa, Se);

        GaussNewton<RealType> GN{};
        GN.set_tolerance(1e-15); GN.set_maximum_iterations(100);

        VectorType x_std, x_n, x_m;
        std.compute(x_std, y, GN);
        nform.compute(x_n, y, GN);
        mform.compute(x_m, y, GN);

        auto e1 = maximum_error(x_std, x_m);
        auto e2 = maximum_error(x_std, x_n);

        ensure_small(e1, "Standard - nform");
        ensure_small(e2, "Standard - nform");

        // Test inversion using CG solver.
        ConjugateGradient<> cg(1e-15);
        GaussNewton<RealType, ConjugateGradient<>> GN_CG(cg);
        GN_CG.set_tolerance(1e-15); GN_CG.set_maximum_iterations(100);

        std.compute(x_std, y, GN_CG);
        nform.compute(x_n, y, GN_CG);
        mform.compute(x_m, y, GN_CG);

        e1 = maximum_error(x_std, x_m);
        e2 = maximum_error(x_std, x_n);

        ensure_small(e1, "Standard - nform CG");
        ensure_small(e2, "Standard - nform CG");
    }
};

TESTMAIN(GenericTest<SphereModel COMMA matrix_types>::run(10);)
