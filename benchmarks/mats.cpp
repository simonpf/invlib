#include "invlib/map.h"
#include "invlib/algebra.h"
#include "invlib/optimization.h"
#include "invlib/io.h"
#include "invlib/profiling/timer.h"
#include "invlib/mkl/mkl_sparse.h"
#include "invlib/interfaces/eigen.h"
#include "benchmark_types.h"

#define STR(arg) #arg
#define STR_VALUE(arg) STR(arg)

using namespace invlib;

template
<
typename MatrixType,
typename VectorType
>
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

template<typename SparseType>
struct Mats
{

    Mats() = default;
    Mats(const Mats & ) = delete;
    Mats(      Mats &&)             = delete;
    Mats & operator=(const Mats & ) = delete;
    Mats & operator=(      Mats &&) = delete;

    std::tuple<double, double, double> operator()()
    {

        reset_times();

        // Define types.
        using MatrixType = Matrix<Timer<SparseType>>;
        using VectorType = Vector<Timer<typename SparseType::VectorType>>;

        using SolverType      = invlib::ConjugateGradient<CGStepLimit<1000>>;
        using MinimizerType   = invlib::GaussNewton<double, SolverType>;
        using PrecisionMatrix = invlib::PrecisionMatrix<MatrixType>;
        using MAPType         = invlib::MAP<LinearModel<MatrixType, VectorType>,
                                            MatrixType,
                                            PrecisionMatrix,
                                            PrecisionMatrix>;

        // Load data.
        MatrixType K(read_matrix_arts(STR_VALUE(MATS_DATA) "/K.xml"));
        MatrixType SaInv(read_matrix_arts(STR_VALUE(MATS_DATA) "/SaInv.xml"));
        MatrixType SeInv(read_matrix_arts(STR_VALUE(MATS_DATA) "/SeInv.xml"));
        VectorType xa(read_vector_arts(STR_VALUE(MATS_DATA) "/xa.xml"));
        VectorType y(read_vector_arts(STR_VALUE(MATS_DATA) "/y.xml"));

        PrecisionMatrix Pa(SaInv);
        PrecisionMatrix Pe(SeInv);

        // Setup OEM.
        SolverType                          cg(1e-6, 0);
        MinimizerType                       gn(1e-6, 1, cg);
        LinearModel<MatrixType, VectorType> F(K, xa);
        MAPType                             oem(F, xa, Pa, Pe);

        // Run OEM.
        VectorType x;

        auto t1 = steady_clock::now();
        oem.compute(x, y, gn, 0);
        auto t2 = steady_clock::now();
        auto oem_time = duration_cast<duration<double>>(t2 - t1);

        double mv_time  = multiply_mv_time.count();
        double mtv_time = multiply_mtv_time.count();

        return std::make_tuple(oem_time.count(), mv_time, mtv_time);
    }
};

template <typename T> void foo();

int main()
{
    using TypeList = typename ConcatTuple<MklSparseTypes, std::tuple<EigenSparse>>::Type;
    Benchmark<Mats, TypeList>().run();
}
