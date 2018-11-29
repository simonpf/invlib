#include <Python.h>
#include "invlib/interfaces/python/python_vector.h"
#include "invlib/interfaces/python/python_matrix.h"
#include "invlib/algebra.h"

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

}
