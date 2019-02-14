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

using ScalarType = @FLOATTYPE@;

extern "C" {

////////////////////////////////////////////////////////////////////////////////
// Vectors
////////////////////////////////////////////////////////////////////////////////

    //
    // Vector creation & destruction
    //

    void* create_vector(void *data,
                        size_t n,
                        bool copy)
    {
        auto v = new PythonVector<ScalarType>(
            invlib::PythonVector<ScalarType>(reinterpret_cast<ScalarType*>(data), n, copy)
            );
        return v;
    }

    void destroy_vector(void *v)
    {
        delete reinterpret_cast<PythonVector<ScalarType>*>(v);
    }

    void * vector_get_data_pointer(void *v) {
        return reinterpret_cast<PythonVector<ScalarType>*>(v)->get_element_pointer();
    }

    size_t vector_rows(void *v) {
        return reinterpret_cast<PythonVector<ScalarType>*>(v)->rows();
    }

    //
    // Vector arithmetic
    //

    ScalarType vector_dot(void *a,
                          void *b)
    {
        auto & a_ = *reinterpret_cast<PythonVector<ScalarType>*>(a);
        auto & b_ = *reinterpret_cast<PythonVector<ScalarType>*>(b);

        return dot(a_, b_);
    }

    void * vector_add(void *a,
                      void *b)
    {
        auto & a_ = *reinterpret_cast<PythonVector<ScalarType>*>(a);
        auto & b_ = *reinterpret_cast<PythonVector<ScalarType>*>(b);
        auto   c_ = new PythonVector<ScalarType>(a_ + b_);

        return c_;
    }

    void * vector_subtract(void *a,
                           void *b)
    {
        auto & a_ = *reinterpret_cast<PythonVector<ScalarType>*>(a);
        auto & b_ = *reinterpret_cast<PythonVector<ScalarType>*>(b);
        auto   c_ = new PythonVector<ScalarType>(a_ - b_);
 
        return c_;
    }

    void vector_scale(void *a,
                      ScalarType c)
    {
        auto & a_ = *reinterpret_cast<PythonVector<ScalarType>*>(a);
        a_.scale(c);
    }

////////////////////////////////////////////////////////////////////////////////
// Dense matrices
////////////////////////////////////////////////////////////////////////////////

    //
    // Matrix creation and destruction
    //

    void* create_matrix(void *data,
                        size_t m,
                        size_t n,
                        bool copy)
    {
        auto A = new PythonMatrix<ScalarType>(
            invlib::PythonMatrix<ScalarType>(reinterpret_cast<ScalarType*>(data), m, n, copy)
            );
        return A;
    }

    void destroy_matrix(void * A)
    {
        delete reinterpret_cast<PythonMatrix<ScalarType> *>(A);
    }

    size_t matrix_rows(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix<ScalarType>*>(A);
        return A_.rows();
    }

    size_t matrix_cols(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix<ScalarType>*>(A);
        return A_.cols();
    }

    void * matrix_get_data_pointer(void *v) {
        return reinterpret_cast<PythonMatrix<ScalarType>*>(v)->get_element_pointer();
    }

    //
    // Arithmetic
    //

    void* matrix_vector_multiply(void *A,
                                 void *u)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix<ScalarType>*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<ScalarType>*>(u);
        auto   v_ = new PythonVector<ScalarType>(A_ * u_);

        return v_;
    }

    void* matrix_vector_multiply_transpose(void *A,
                                           void *u)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix<ScalarType>*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<ScalarType>*>(u);
        auto   v_ = new PythonVector<ScalarType>(transp(A_) * u_);
        return v_;
    }

    void* matrix_matrix_multiply(void *A,
                                 void *B)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix<ScalarType>*>(A);
        auto & B_ = *reinterpret_cast<PythonMatrix<ScalarType>*>(B);
        auto   C_ = new PythonMatrix<ScalarType>(A_ * B_);
        return C_;
    }

////////////////////////////////////////////////////////////////////////////////
// Sparse MKL matrices
////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // CSR Format
    ////////////////////////////////////////////////////////////////////////////////

    using invlib::Representation;

    using SparseDataCsr = invlib::SparseData<ScalarType,
                                             MKL_INT,
                                             Representation::CompressedRows>;
    using MklCsr = invlib::Matrix<invlib::MklSparse<ScalarType,
                                                    Representation::CompressedRows>>;

    void* create_sparse_mkl_csr(MKL_INT    m,
                                MKL_INT    n,
                                MKL_INT    nnz,
                                MKL_INT   *row_starts,
                                MKL_INT   *column_indices,
                                ScalarType *elements,
                                bool copy)
    {
        // We need to copy the data from the Python pointers as MKL requires
        // an extra array pointing to the end of every column.

        // row starts
        std::shared_ptr<MKL_INT *> row_starts_;
        if (copy) {
            row_starts_ = std::shared_ptr<MKL_INT *>(new (MKL_INT *),
                                                     invlib::ArrayDeleter<MKL_INT *>());
            *row_starts_ = new MKL_INT[m + 1];
            std::copy(row_starts, row_starts + m, *row_starts_);
            (*row_starts_)[m] = nnz;
        } else {
            row_starts_ = std::shared_ptr<MKL_INT *>(new (MKL_INT *));
        }


        // column indices
        std::shared_ptr<MKL_INT *> column_indices_;
        if (copy) {
            column_indices_ = std::shared_ptr<MKL_INT *>(new (MKL_INT *),
                                                         invlib::ArrayDeleter<MKL_INT *>());
            *column_indices_ = new MKL_INT[nnz];
            std::copy(column_indices, column_indices + nnz, *column_indices_);
        } else {
            column_indices_ = std::shared_ptr<MKL_INT *>(new (MKL_INT *));
        }


        // elements
        std::shared_ptr<ScalarType *> elements_;
        if (copy) {
            elements_ = std::shared_ptr<ScalarType *>(new (ScalarType *),
                                                      invlib::ArrayDeleter<ScalarType *>());
            (*elements_) = new ScalarType[nnz];
            std::copy(elements, elements + nnz, *elements_);
        } else {
            elements_ = std::shared_ptr<ScalarType *>(new (ScalarType *));
        }


        auto data = SparseDataCsr(m, n, nnz,
                                  row_starts_,
                                  column_indices_,
                                  elements_);
        auto A    = new MklCsr(data);
        return A;
    }

    void* sparse_mkl_csr_multiply(void * A,
                                  void * u)
    {
        auto & A_ = *reinterpret_cast<MklCsr*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<ScalarType>*>(u);
        auto v_ = new PythonVector<ScalarType>(A_ * u_);

        return v_;
    }

    void* sparse_mkl_csr_transpose_multiply(void * A,
                                            void * u)
    {
        auto & A_ = *reinterpret_cast<MklCsr*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<ScalarType>*>(u);

        auto   v_ = new PythonVector<ScalarType>(transp(A_) * u_);

        return v_;
    }

    using SparseDataCsr = invlib::SparseData<ScalarType,
                                             MKL_INT,
                                             Representation::CompressedRows>;
    using MklCsr = invlib::Matrix <invlib::MklSparse<ScalarType,
                                                     Representation::CompressedRows>>;

    ////////////////////////////////////////////////////////////////////////////////
    // CSC Format
    ////////////////////////////////////////////////////////////////////////////////

    using invlib::Representation;

    using SparseDataCsc = invlib::SparseData<ScalarType,
                                             MKL_INT,
                                             Representation::CompressedColumns>;
    using MklCsc = invlib::Matrix<invlib::MklSparse<ScalarType,
                                                    Representation::CompressedColumns>>;

    void* create_sparse_mkl_csc(MKL_INT    m,
                                MKL_INT    n,
                                MKL_INT    nnz,
                                MKL_INT    *column_starts,
                                MKL_INT    *row_indices,
                                ScalarType *elements,
                                bool copy)
    {
        // We need to copy the data from the Python pointers as MKL requires
        // an extra array pointing to the end of every column.

        // row starts
        std::shared_ptr<MKL_INT *> column_starts_;
        if (copy) {
            column_starts_ = std::shared_ptr<MKL_INT *>(new (MKL_INT *),
                                                     invlib::ArrayDeleter<MKL_INT *>());
            *column_starts_ = new MKL_INT[n + 1];
            std::copy(column_starts, column_starts + n, *column_starts_);
            (*column_starts_)[n] = nnz;
        } else {
            column_starts_ = std::shared_ptr<MKL_INT *>(new (MKL_INT *));
            (*column_starts_) = column_starts;
        }


        // column indices
        std::shared_ptr<MKL_INT *> row_indices_;
        if (copy) {
            row_indices_ = std::shared_ptr<MKL_INT *>(new (MKL_INT *),
                                                      invlib::ArrayDeleter<MKL_INT *>());
            *row_indices_ = new MKL_INT[nnz];
            std::copy(row_indices, row_indices + nnz, *row_indices_);
        } else {
            row_indices_ = std::shared_ptr<MKL_INT *>(new (MKL_INT *));
            (*row_indices_) = row_indices;
        }


        // elements
        std::shared_ptr<ScalarType *> elements_;
        if (copy) {
            elements_ = std::shared_ptr<ScalarType *>(new (ScalarType *),
                                                 invlib::ArrayDeleter<ScalarType *>());
            (*elements_) = new ScalarType[nnz];
            std::copy(elements, elements + nnz, *elements_);
        } else {
            elements_ = std::shared_ptr<ScalarType *>(new (ScalarType *));
            (*elements_) = elements;
        }


        auto data = SparseDataCsc(m, n, nnz,
                                  row_indices_,
                                  column_starts_,
                                  elements_);
        auto A    = new MklCsc(data);
        return A;
    }

    void* sparse_mkl_csc_multiply(void * A,
                                  void * u)
    {
        auto & A_ = *reinterpret_cast<MklCsc*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<ScalarType>*>(u);
        auto v_ = new PythonVector<ScalarType>(A_ * u_);

        return v_;
    }

    void* sparse_mkl_csc_transpose_multiply(void * A,
                                            void * u)
    {
        auto & A_ = *reinterpret_cast<MklCsc*>(A);
        auto & u_ = *reinterpret_cast<PythonVector<ScalarType>*>(u);

        auto   v_ = new PythonVector<ScalarType>(transp(A_) * u_);

        return v_;
    }


////////////////////////////////////////////////////////////////////////////////
// OEM
////////////////////////////////////////////////////////////////////////////////

    using SolverType = invlib::ConjugateGradient<>;

    using Minimizer = invlib::GaussNewton<ScalarType, SolverType>;
    using Linear    = invlib::LinearModel<MklCsc>;
    using Covmat    = invlib::PrecisionMatrix<MklCsc>;
    using LinearMAP = invlib::MAP<Linear, MklCsc, Covmat,
                                  Covmat, PythonVector<ScalarType>>;

    void * map_linear(void * K,
                      void * SaInv,
                      void * SeInv,
                      void * x_a,
                      void * y)
    {
        Covmat SaInv_(*reinterpret_cast<MklCsc*>(SaInv));
        Covmat SeInv_(*reinterpret_cast<MklCsc*>(SeInv));

        auto & x_a_ = *reinterpret_cast<PythonVector<ScalarType> *>(x_a);
        auto & y_  = *reinterpret_cast<PythonVector<ScalarType> *>(y);

        Linear forward_model(*reinterpret_cast<MklCsc*>(K));


        SolverType cg = SolverType(1e-6, 1);
        Minimizer  gn(1e-6, 1, cg);
        LinearMAP  oem(forward_model, x_a_, SaInv_, SeInv_);

        auto x_ = new PythonVector<ScalarType>{};

        oem.compute(*x_, y_, gn, 1);

        return x_;
    }

    void * forward_model_linear(void * K,
                                void * x)
    {
        std::cout << K << " // " << x << std::endl;
        auto & x_ = *reinterpret_cast<PythonVector<ScalarType> *>(x);
        Linear forward_model(*reinterpret_cast<MklCsc*>(K));
        auto y_ = new PythonVector<ScalarType>{forward_model.evaluate(x_)};
        return y_;
    }

    void * covmat_multiply(void * S,
                           void * x)
    {
        Covmat S_(*reinterpret_cast<MklCsc*>(S));
        auto & x_ = *reinterpret_cast<PythonVector<ScalarType> *>(x);
        auto y_ = new PythonVector<ScalarType>(inv(S_) * x_);
        return y_;
    }
}


