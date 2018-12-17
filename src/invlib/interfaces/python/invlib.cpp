#include <Python.h>
#include "invlib/interfaces/python/python_vector.h"
#include "invlib/interfaces/python/python_matrix.h"
#include "invlib/optimization/gauss_newton.h"
#include "invlib/map.h"
#include "invlib/forward_models.h"
#include "invlib/mkl/mkl_sparse.h"
#include "invlib/algebra.h"
#include "mkl.h"

#define DLL_PUBLIC __attribute__ ((visibility ("default")))

template<typename ScalarType>
using PythonVector = invlib::Vector<invlib::PythonVector<ScalarType>>;

template<typename ScalarType>
using PythonMatrix = invlib::Matrix<invlib::PythonMatrix<ScalarType>>;

extern "C" {

////////////////////////////////////////////////////////////////////////////////
// Vectors
////////////////////////////////////////////////////////////////////////////////

    //
    // Vector creation & destruction
    //

    void* create_vector_float(void *data,
                              size_t n,
                              bool copy)
    {
        auto v = new PythonVector<float>(
            invlib::PythonVector<float>(reinterpret_cast<float*>(data), n, copy)
            );
        return v;
    }

    void* create_vector_double(void *data,
                              size_t n,
                              bool copy)
    {
        auto v = new PythonVector<double>(
            invlib::PythonVector<double>(reinterpret_cast<double*>(data), n, copy)
            );
        return v;
    }

    void destroy_vector_float(void *v)
    {
        delete reinterpret_cast<PythonVector<float>*>(v);
    }

    void destroy_vector_double(void *v)
    {
        delete reinterpret_cast<PythonVector<double>*>(v);
    }

    void * vector_get_data_pointer_float(void *v) {
        return reinterpret_cast<PythonVector<float>*>(v)->get_element_pointer();
    }

    void * vector_get_data_pointer_double(void *v) {
        return reinterpret_cast<PythonVector<double>*>(v)->get_element_pointer();
    }

    size_t vector_rows_float(void *v) {
        return reinterpret_cast<PythonVector<float>*>(v)->rows();
    }

    size_t vector_rows_double(void *v) {
        return reinterpret_cast<PythonVector<double>*>(v)->rows();
    }

    //
    // Vector arithmetic
    //

    float vector_dot_float(void *a,
                           void *b)
    {
        auto & a_ = *reinterpret_cast<PythonVector<float>*>(a);
        auto & b_ = *reinterpret_cast<PythonVector<float>*>(b);

        return dot(a_, b_);
    }

    double vector_dot_double(void *a,
                             void *b)
    {
        auto & a_ = *reinterpret_cast<PythonVector<double>*>(a);
        auto & b_ = *reinterpret_cast<PythonVector<double>*>(b);

        return dot(a_, b_);
    }

    void * vector_add_float(void *a,
                            void *b)
    {
        auto & a_ = *reinterpret_cast<PythonVector<float>*>(a);
        auto & b_ = *reinterpret_cast<PythonVector<float>*>(b);
        auto   c_ = new PythonVector<float>(a_ + b_);

        return c_;
    }

    void * vector_add_double(void *a,
                             void *b)
    {
        auto & a_ = *reinterpret_cast<PythonVector<double>*>(a);
        auto & b_ = *reinterpret_cast<PythonVector<double>*>(b);
        auto c_ = new PythonVector<double>(a_ + b_);

        return c_;
    }

    void * vector_subtract_float(void *a,
                                 void *b)
    {
        auto & a_ = *reinterpret_cast<PythonVector<float>*>(a);
        auto & b_ = *reinterpret_cast<PythonVector<float>*>(b);
        auto   c_ = new PythonVector<float>(a_ - b_);

        return c_;
    }

    void * vector_subtract_double(void *a,
                                  void *b)
    {
        auto & a_ = *reinterpret_cast<PythonVector<double>*>(a);
        auto & b_ = *reinterpret_cast<PythonVector<double>*>(b);
        auto   c_ = new PythonVector<double>(a_ - b_);

        return c_;
    }

    void vector_scale_float(void *a,
                            float c)
    {
        auto & a_ = *reinterpret_cast<PythonVector<float>*>(a);
        a_.scale(c);
    }

    void vector_scale_double(void *a,
                             double c)
    {
        auto & a_ = *reinterpret_cast<PythonVector<double>*>(a);
        a_.scale(c);
    }

////////////////////////////////////////////////////////////////////////////////
// Dense matrices
////////////////////////////////////////////////////////////////////////////////

    //
    // Matrix creation and destruction
    //

    void* create_matrix_float(void *data,
                              size_t m,
                              size_t n,
                              bool copy)
    {
        auto A = new PythonMatrix<float>(
            invlib::PythonMatrix<float>(reinterpret_cast<float*>(data), m, n, copy)
            );
        return A;
    }

    void* create_matrix_double(void *data,
                               size_t m,
                               size_t n,
                               bool copy)
    {
        auto A = new PythonMatrix<double>(
            invlib::PythonMatrix<double>(reinterpret_cast<double*>(data), m, n, copy)
            );
        return A;
    }

    void destroy_matrix_float(void * A)
    {
        delete reinterpret_cast<PythonMatrix<float> *>(A);
    }

    void destroy_matrix_double(void *A)
    {
        delete reinterpret_cast<PythonMatrix<double>*>(A);
    }

    size_t matrix_rows_float(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix<float>*>(A);
        return A_.rows();
    }

    size_t matrix_rows_double(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix<double>*>(A);
        return A_.rows();
    }

    size_t matrix_cols_float(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix<float>*>(A);
        return A_.cols();
    }

    size_t matrix_cols_double(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix<double>*>(A);
        return A_.cols();
    }

    void * matrix_get_data_pointer_float(void *v) {
        return reinterpret_cast<PythonMatrix<float>*>(v)->get_element_pointer();
    }

    void * matrix_get_data_pointer_double(void *v) {
        return reinterpret_cast<PythonMatrix<double>*>(v)->get_element_pointer();
    }

    //
    // Arithmetic
    //

    void* matrix_vector_multiply_float(void *A,
                                       void *u)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix<float>*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<float>*>(u);
        auto   v_ = new PythonVector<float>(A_ * u_);

        return v_;
    }

    void* matrix_vector_multiply_double(void *A,
                                        void *u)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix<double>*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<double>*>(u);
        auto   v_ = new PythonVector<double>(A_ * u_);

        return v_;
    }

    void* matrix_vector_multiply_transpose_float(void *A,
                                                 void *u)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix<float>*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<float>*>(u);
        auto   v_ = new PythonVector<float>(transp(A_) * u_);
        return v_;
    }

    void* matrix_vector_multiply_transpose_double(void *A,
                                                  void *u)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix<double>*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<double>*>(u);
        auto   v_ = new PythonVector<double>(transp(A_) * u_);
        return v_;
    }

    void* matrix_matrix_multiply_float(void *A,
                                       void *B)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix<float>*>(A);
        auto & B_ = *reinterpret_cast<PythonMatrix<float>*>(B);
        auto   C_ = new PythonMatrix<float>(A_ * B_);
        return C_;
    }

    void* matrix_matrix_multiply_double(void *A,
                                        void *B)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix<double>*>(A);
        auto & B_ = *reinterpret_cast<PythonMatrix<double>*>(B);
        auto   C_ = new PythonMatrix<double>(A_ * B_);
        return C_;
    }

////////////////////////////////////////////////////////////////////////////////
// Sparse MKL matrices
////////////////////////////////////////////////////////////////////////////////

    using invlib::Representation;

    using SparseDataCsrFloat = invlib::SparseData<float,
                                                  MKL_INT,
                                                  Representation::CompressedRows>;
    using MklCsrFloat = invlib::Matrix<invlib::MklSparse<float,
                                                         Representation::CompressedRows>>;

    void* create_sparse_mkl_csr_float(MKL_INT    m,
                                      MKL_INT    n,
                                      MKL_INT    nnz,
                                      MKL_INT   *row_starts,
                                      MKL_INT   *column_indices,
                                      float *elements)
    {
        // We need to copy the data from the Python pointers as MKL requires
        // an extra array pointing to the end of every column.

        // row starts
        auto row_starts_     = std::shared_ptr<MKL_INT *>(new (MKL_INT *),
                                                          invlib::ArrayDeleter<MKL_INT *>());
        *row_starts_ = new MKL_INT[m + 1];
        std::copy(row_starts, row_starts + m, *row_starts_);
        (*row_starts_)[m] = nnz;

        // column indices
        auto column_indices_ = std::shared_ptr<MKL_INT *>(new (MKL_INT *),
                                                          invlib::ArrayDeleter<MKL_INT *>());
        *column_indices_ = new MKL_INT[nnz];
        std::copy(column_indices, column_indices + nnz, *column_indices_);

        // elements
        auto elements_       = std::shared_ptr<float *>(new (float *),
                                                         invlib::ArrayDeleter<float *>());
        (*elements_) = new float[nnz];
        std::copy(elements, elements + nnz, *elements_);


        auto data = SparseDataCsrFloat(m, n, nnz,
                                       row_starts_,
                                       column_indices_,
                                       elements_);
        auto A    = new MklCsrFloat(data);
        return A;
    }

    void* sparse_mkl_csr_multiply_float(void * A,
                                        void * u)
    {
        auto & A_ = *reinterpret_cast<MklCsrFloat*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<float>*>(u);
        auto v_ = new PythonVector<float>(A_ * u_);

        return v_;
    }

    void* sparse_mkl_csr_transpose_multiply_float(void * A,
                                                  void * u)
    {
        auto & A_ = *reinterpret_cast<MklCsrFloat*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<float>*>(u);

        auto   v_ = new PythonVector<float>(transp(A_) * u_);

        return v_;
    }

    using SparseDataCsrDouble = invlib::SparseData<double,
                                                   MKL_INT,
                                                   Representation::CompressedRows>;
    using MklCsrDouble = invlib::Matrix <invlib::MklSparse<double,
                                                           Representation::CompressedRows>>;

    void* create_sparse_mkl_csr_double(MKL_INT    m,
                                       MKL_INT    n,
                                       MKL_INT    nnz,
                                       MKL_INT   *row_starts,
                                       MKL_INT   *column_indices,
                                       double    *elements)
    {
        // We need to copy the data from the Python pointers as MKL requires
        // an extra array pointing to the end of every column.

        // row starts
        auto row_starts_     = std::shared_ptr<MKL_INT *>(new (MKL_INT *),
                                                          invlib::ArrayDeleter<MKL_INT *>());
        *row_starts_ = new MKL_INT[m + 1];
        std::copy(row_starts, row_starts + m, *row_starts_);
        (*row_starts_)[m] = nnz;

        // column indices
        auto column_indices_ = std::shared_ptr<MKL_INT *>(new (MKL_INT *),
                                                          invlib::ArrayDeleter<MKL_INT *>());
        *column_indices_ = new MKL_INT[nnz];
        std::copy(column_indices, column_indices + nnz, *column_indices_);

        // elements
        auto elements_       = std::shared_ptr<double *>(new (double *),
                                                         invlib::ArrayDeleter<double *>());
        (*elements_) = new double[nnz];
        std::copy(elements, elements + nnz, *elements_);


        auto data = SparseDataCsrDouble(m, n, nnz,
                                        row_starts_,
                                        column_indices_,
                                        elements_);
        auto A    = new MklCsrDouble(data);
        return A;
    }

    void* sparse_mkl_csr_multiply_double(void * A,
                                         void * u)
    {
        auto & A_ = *reinterpret_cast<MklCsrDouble*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<double>*>(u);

        auto   v_ = new PythonVector<double>(A_ * u_);

        return v_;
    }

    void* sparse_mkl_csr_transpose_multiply_double(void * A,
                                                   void * u)
    {
        auto & A_ = *reinterpret_cast<MklCsrDouble*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<double>*>(u);

        auto   v_ = new PythonVector<double>(transp(A_) * u_);

        return v_;
    }

    void* sparse_mkl_csr_solve_double(void * A,
                                      void * u)
    {
        auto & A_ = *reinterpret_cast<MklCsrDouble*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<double>*>(u);

        auto   v_ = new PythonVector<double>(transp(A_) * u_);

        return v_;
    }

////////////////////////////////////////////////////////////////////////////////
// OEM
////////////////////////////////////////////////////////////////////////////////

    using SolverType = invlib::ConjugateGradient<>;

    using MinimizerDouble = invlib::GaussNewton<double, SolverType>;
    using LinearDouble    = invlib::LinearModel<MklCsrDouble>;
    using CovmatDouble    = invlib::PrecisionMatrix<MklCsrDouble>;
    using LinearMAPDouble = invlib::MAP<LinearDouble,
                                        MklCsrDouble,
                                        CovmatDouble,
                                        CovmatDouble,
                                        PythonVector<double>>;

    void * map_linear_double(void * K,
                             void * SaInv,
                             void * SeInv,
                             void * xa,
                             void * y)
    {
        LinearDouble forward_model(*reinterpret_cast<MklCsrDouble*>(K));
        CovmatDouble SaInv_(*reinterpret_cast<MklCsrDouble*>(SaInv));
        CovmatDouble SeInv_(*reinterpret_cast<MklCsrDouble*>(SeInv));

        auto & xa_ = *reinterpret_cast<PythonVector<double> *>(xa);
        auto & y_  = *reinterpret_cast<PythonVector<double> *>(y);

        SolverType      cg = SolverType(1e-6);
        MinimizerDouble gn(1e-6, 1, cg);
        LinearMAPDouble oem(forward_model, xa_, SaInv_, SeInv_);

        auto & x_ = *(new PythonVector<double>{});
        x_.resize(forward_model.n);

        oem.compute(x_, y_, gn);

        return &x_;
    }

}


