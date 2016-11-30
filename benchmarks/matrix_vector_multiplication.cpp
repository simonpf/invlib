#include <iomanip>

#include "invlib/algebra.h"
#include "invlib/profiling/timer.h"
#include "benchmark_types.h"

using namespace invlib;

template <typename SparseType>
struct MatrixVectorMultiplication
{
    auto operator()()
        -> decltype(vector_to_tuple<20>(std::vector<double>()))
    {

        using MatrixType = Matrix<Timer<SparseType>>;
        using VectorType = Vector<Timer<typename SparseType::VectorType>>;
        using RealType   = typename SparseType::RealType;
        using SparseDataType = SparseData<RealType, int, Representation::Coordinates>;
        using VectorDataType = VectorData<RealType>;

        std::vector<double> results{};
        results.reserve(20);

        for (size_t i = 1; i < 11; i++)
        {
            duration<double> mv_time  = duration<double>::zero();
            duration<double> mtv_time = duration<double>::zero();

            size_t m = i * 100000;
            size_t n = i * 100000;
            auto A_data = SparseDataType::random(m, n);
            auto v_data = VectorDataType::random(n);

            for (size_t j = 1; j < 10; j++)
            {

            MatrixType A(A_data);
            VectorType v(v_data);

            // Matrix Vector Multiplication
            auto t1 = steady_clock::now();
            VectorDataType w  = A * A * A * A * A * v;
            auto t2 = steady_clock::now();
            mv_time += duration_cast<duration<double>>(t2 - t1);

            // Transposed Matrix Vector Multiplication
            auto At = transp(A);
            t1 = steady_clock::now();
            VectorDataType wt  = At * At * At * At * At * v;
            t2 = steady_clock::now();
            mtv_time += duration_cast<duration<double>>(t2 - t1);
            }

            results.push_back(mv_time.count() / 10.0);
            results.push_back(mtv_time.count() / 10.0);
        }
        return vector_to_tuple<20>(results);
    }
};

int main()
{
    Benchmark<MatrixVectorMultiplication, MklSparseTypes>().run();
}
