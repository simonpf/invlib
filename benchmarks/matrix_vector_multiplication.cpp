#include <iomanip>

#include "invlib/algebra.h"
#include "invlib/profiling/timer.h"
#include "invlib/cuda/cuda_sparse.h"
#include "invlib/mkl/mkl_sparse.h"
#include "invlib/interfaces/eigen.h"

using namespace invlib;

void run_benchmark()
{
    for (size_t i = 1; i < 20; i++)
    {
        std::cout << "n = " << 100000 * i << std::endl;

        size_t m = i * 100000;
        size_t n = i * 100000;
        auto A = SparseData<double, int, Representation::Coordinates>::random(m, n);
        auto v = VectorData<double>::random(n);

        for (size_t j = 1; j < 10; j++)
        {
            using MklMatrixType
                = Matrix<Timer<MklSparse<double, Representation::Coordinates>>>;
            using MklVectorType = Matrix<Timer<BlasVector<double>>>;

            MklMatrixType A_mkl(A);
            MklVectorType v_mkl(v);
            MklVectorType w_mkl  = A_mkl * A_mkl * A_mkl * A_mkl * A_mkl * v_mkl;
            auto At_mkl = transp(A_mkl);
            MklVectorType wt_mkl  = At_mkl * At_mkl * At_mkl * At_mkl * At_mkl * v_mkl;
        }
        std::cout << "MKL, coo  "   << std::setw(15) << multiply_mv_time.count() / 10.0;
        std::cout << std::setw(15) << multiply_mtv_time.count() / 10.0 << std::endl;

        multiply_mv_time = std::chrono::duration<double>::zero();
        multiply_mtv_time = std::chrono::duration<double>::zero();

        for (size_t j = 1; j < 10; j++)
        {
            using MklMatrixType
                = Matrix<Timer<MklSparse<double, Representation::Coordinates>>>;
            using MklVectorType = Matrix<Timer<BlasVector<double>>>;

            MklMatrixType A_mkl(A);
            MklVectorType v_mkl(v);
            MklVectorType w_mkl  = A_mkl * A_mkl * A_mkl * A_mkl * A_mkl * v_mkl;
            auto At_mkl = transp(A_mkl);
            MklVectorType wt_mkl  = At_mkl * At_mkl * At_mkl * At_mkl * At_mkl * v_mkl;
        }

        std::cout << "MKL, csc  "   << std::setw(15) << multiply_mv_time.count() / 10.0;
        std::cout << std::setw(15) << multiply_mtv_time.count() / 10.0 << std::endl;

        multiply_mv_time = std::chrono::duration<double>::zero();
        multiply_mtv_time = std::chrono::duration<double>::zero();

        for (size_t j = 1; j < 10; j++)
        {
            using MklMatrixType
                = Matrix<Timer<MklSparse<double, Representation::CompressedRows>>>;
            using MklVectorType = Matrix<Timer<BlasVector<double>>>;

            MklMatrixType A_mkl(A);
            MklVectorType v_mkl(v);
            MklVectorType w_mkl  = A_mkl * A_mkl * A_mkl * A_mkl * A_mkl * v_mkl;
            auto At_mkl = transp(A_mkl);
            MklVectorType wt_mkl  = At_mkl * At_mkl * At_mkl * At_mkl * At_mkl * v_mkl;
        }

        std::cout << "MKL, csr  "   << std::setw(15) << multiply_mv_time.count() / 10.0;
        std::cout << std::setw(15) << multiply_mtv_time.count() / 10.0 << std::endl;

        multiply_mv_time = std::chrono::duration<double>::zero();
        multiply_mtv_time = std::chrono::duration<double>::zero();

        for (size_t j = 1; j < 10; j++)
        {
            using MklMatrixType
                = Matrix<Timer<MklSparse<double, Representation::Hybrid>>>;
            using MklVectorType = Matrix<Timer<BlasVector<double>>>;

            MklMatrixType A_mkl(A);
            MklVectorType v_mkl(v);
            MklVectorType w_mkl  = A_mkl * A_mkl * A_mkl * A_mkl * A_mkl * v_mkl;
            auto At_mkl = transp(A_mkl);
            MklVectorType wt_mkl  = At_mkl * At_mkl * At_mkl * At_mkl * At_mkl * v_mkl;
        }

        std::cout << "MKL, hyb  "   << std::setw(15) << multiply_mv_time.count() / 10.0;
        std::cout << std::setw(15) << multiply_mtv_time.count() / 10.0 << std::endl;

        multiply_mv_time = std::chrono::duration<double>::zero();
        multiply_mtv_time = std::chrono::duration<double>::zero();

        for (size_t j = 1; j < 10; j++)
        {
            using EigenMatrixType
                = Matrix<Timer<EigenSparse>>;
            using EigenVectorType = Matrix<Timer<EigenVector>>;

            EigenMatrixType A_eigen(A);
            EigenVectorType v_eigen(v);
            EigenVectorType w_eigen  = A_eigen * A_eigen * A_eigen * A_eigen * A_eigen * v_eigen;
            auto At_eigen = transp(A_eigen);
            EigenVectorType wt_eigen  = At_eigen * At_eigen * At_eigen * At_eigen * At_eigen * v_eigen;
        }

        std::cout << "Eigen    "   << std::setw(15) << multiply_mv_time.count() / 10.0;
        std::cout << std::setw(15) << multiply_mtv_time.count() / 10.0 << std::endl;
        std::cout << std::endl;

        multiply_mv_time = std::chrono::duration<double>::zero();
        multiply_mtv_time = std::chrono::duration<double>::zero();

//         {
//             using CudaMatrixType
//                 = Matrix<CudaSparse<double, Representation::CompressedRows>>;
//             using CudaVectorType = Matrix<CudaVector<double>>;

//             CudaMatrixType A_cuda(A);
//             CudaVectorType v_cuda(v);


//             auto t1 = steady_clock::now();
//             VectorData<double> w_cuda  =
//                 static_cast<CudaVectorType>(A_cuda * A_cuda * A_cuda * A_cuda * A_cuda * v_cuda);
//             auto t2 = steady_clock::now();
//             multiply_mv_time = duration_cast<duration<double>>(t2 - t1);

//             t1 = steady_clock::now();
//             auto At_cuda = transp(A_cuda);
//             VectorData<double> wt_mkl  =
//                 static_cast<CudaVectorType>(At_cuda * At_cuda * At_cuda * At_cuda * At_cuda * v_cuda);
//             t2 = steady_clock::now();
//             multiply_mtv_time = duration_cast<duration<double>>(t2 - t1);
//         }

//         std::cout << "CUDA, csr "   << std::setw(15) << multiply_mv_time.count() / 10.0;
//         std::cout << std::setw(15) << multiply_mtv_time.count() / 10.0 << std::endl;

//         multiply_mv_time = std::chrono::duration<double>::zero();
//         multiply_mtv_time = std::chrono::duration<double>::zero();

//         {
//             using CudaMatrixType
//                 = Matrix<CudaSparse<double, Representation::CompressedColumns>>;
//             using CudaVectorType = Matrix<CudaVector<double>>;

//             CudaMatrixType A_cuda(A);
//             CudaVectorType v_cuda(v);


//             auto t1 = steady_clock::now();
//             VectorData<double> w_cuda  =
//                 static_cast<CudaVectorType>(A_cuda * A_cuda * A_cuda * A_cuda * A_cuda * v_cuda);
//             auto t2 = steady_clock::now();
//             multiply_mv_time = duration_cast<duration<double>>(t2 - t1);

//             t1 = steady_clock::now();
//             auto At_cuda = transp(A_cuda);
//             VectorData<double> wt_mkl  =
//                 static_cast<CudaVectorType>(At_cuda * At_cuda * At_cuda * At_cuda * At_cuda * v_cuda);
//             t2 = steady_clock::now();
//             multiply_mtv_time = duration_cast<duration<double>>(t2 - t1);
//         }

//         std::cout << "CUDA, csc "   << std::setw(15) << multiply_mv_time.count() / 10.0;
//         std::cout << std::setw(15) << multiply_mtv_time.count() / 10.0 << std::endl;
        std::cout << std::endl;

        multiply_mv_time = std::chrono::duration<double>::zero();
        multiply_mtv_time = std::chrono::duration<double>::zero();
    }
}

int main()
{
    run_benchmark();
}
