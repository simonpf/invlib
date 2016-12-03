#include <iomanip>

#include "invlib/algebra.h"
#include "invlib/profiling/timer.h"
#include "benchmark_types.h"

using namespace invlib;

template <typename SparseType>
struct MatrixVectorMultiplication
{
    auto operator()()
        -> decltype(vector_to_tuple<40>(std::vector<double>()))
    {

        using MatrixType = Matrix<Timer<SparseType>>;
        using VectorType = Vector<Timer<typename SparseType::VectorType>>;
        using RealType   = typename SparseType::RealType;
        using SparseDataType = SparseData<RealType, int, Representation::Coordinates>;
        using VectorDataType = VectorData<RealType>;

        std::vector<double> results{};
        results.reserve(40);

        for (size_t i = 1; i < 11; i++)
        {
            duration<double> mv_time  = duration<double>::zero();
            duration<double> mtv_time = duration<double>::zero();
            duration<double> dt;
            double mv_time_2  = 0.0;
            double mtv_time_2 = 0.0;

            size_t m = i * 100000;
            size_t n = i * 100000;
            auto A_data = SparseDataType::random(m, n);
            auto v_data = VectorDataType::random(n);

            for (size_t j = 1; j < 100; j++)
            {
                MatrixType A(A_data);
                VectorType v(v_data);

                // Matrix Vector Multiplication
                auto t1 = steady_clock::now();
                VectorDataType w  = static_cast<VectorType>(A * A * A * A * A * v);
                auto t2 = steady_clock::now();
                dt       = duration_cast<duration<double>>(t2 - t1);
                mv_time   += dt;
                mv_time_2 += dt.count() * dt.count();

                // Transposed Matrix Vector Multiplication
                auto At = transp(A);
                t1 = steady_clock::now();
                VectorDataType wt  = static_cast<VectorType>(At * At * At * At * At * v);
                t2 = steady_clock::now();
                dt       = duration_cast<duration<double>>(t2 - t1);
                mtv_time   += dt;
                mtv_time_2 += dt.count() * dt.count();
            }

            results.push_back(mv_time.count() / 100.0);
            results.push_back(sqrt(mv_time_2 / 100.0 - results.back() * results.back()));
            results.push_back(mtv_time.count() / 100.0);
            results.push_back(sqrt(mtv_time_2 / 100.0 - results.back() * results.back()));
        }
        return vector_to_tuple<40>(results);
    }
};

int main()
{
    Benchmark<MatrixVectorMultiplication, SparseTypes>().run();
}
