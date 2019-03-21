#include "invlib/algebra.h"
#include "invlib/map.h"
#include "invlib/forward_models.h"
#include "invlib/optimization/gauss_newton.h"
#include "invlib/utility/array.h"

#include "invlib/interfaces/python/python_matrix.h"
#include "invlib/interfaces/python/python_solver.h"
#include "invlib/interfaces/python/python_forward_model.h"

#include "mkl.h"
#include "invlib/mkl/mkl_sparse.h"

#define DLL_PUBLIC __attribute__ ((visibility ("default")))

using ScalarType = @FLOATTYPE@;
using IndexType  = MKL_INT;

using PythonVector = invlib::Vector<invlib::BlasVector<ScalarType>>;
using PythonMatrix = invlib::Matrix<invlib::PythonMatrix<ScalarType, IndexType>>;

////////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////////

extern "C" {

////////////////////////////////////////////////////////////////////////////////
// Vectors
////////////////////////////////////////////////////////////////////////////////

    //
    // Vector creation & destruction
    //

    void* create_vector(size_t n,
                        void *data,
                        bool copy)
    {
        auto data_ = invlib::array::make_shared<ScalarType>(data, n, copy);
        return new PythonVector(invlib::VectorData<ScalarType>(n, data_));
    }

    void destroy_vector(void *v)
    {
        delete reinterpret_cast<PythonVector*>(v);
    }

    void * vector_element_pointer(void *v) {
        return reinterpret_cast<PythonVector*>(v)->get_element_pointer();
    }

    size_t vector_rows(void *v) {
        return reinterpret_cast<PythonVector*>(v)->rows();
    }

    //
    // Vector arithmetic
    //

    ScalarType vector_dot(void *a,
                          void *b)
    {
        auto & a_ = *reinterpret_cast<PythonVector*>(a);
        auto & b_ = *reinterpret_cast<PythonVector*>(b);

        return dot(a_, b_);
    }

    void * vector_add(void *a,
                      void *b)
    {
        auto & a_ = *reinterpret_cast<PythonVector*>(a);
        auto & b_ = *reinterpret_cast<PythonVector*>(b);
        auto   c_ = new PythonVector(a_ + b_);

        return c_;
    }

    void * vector_subtract(void *a,
                           void *b)
    {
        auto & a_ = *reinterpret_cast<PythonVector*>(a);
        auto & b_ = *reinterpret_cast<PythonVector*>(b);
        auto   c_ = new PythonVector(a_ - b_);

        return c_;
    }

    void vector_scale(void *a,
                      ScalarType c)
    {
        auto & a_ = *reinterpret_cast<PythonVector*>(a);
        a_.scale(c);
    }

////////////////////////////////////////////////////////////////////////////////
// Dense matrices
////////////////////////////////////////////////////////////////////////////////

    //
    // Matrix creation and destruction
    //

    using DataPtr   = std::shared_ptr<ScalarType[]>;
    using DataPtrs  = std::vector<DataPtr>;

    using IndexPtr  = std::shared_ptr<IndexType[]>;
    using IndexPtrs = std::vector<IndexPtr>;

    void* create_matrix(size_t m,
                        size_t n,
                        size_t nnz,
                        void **index_ptrs,
                        void **data_ptrs,
                        unsigned format,
                        bool copy)
    {
        DataPtrs  data_ptrs_{};
        IndexPtrs index_ptrs_{};

        auto format_ = static_cast<invlib::Format>(format);
        switch (format_) {
        case invlib::Format::Dense : {
            data_ptrs_.push_back(invlib::array::make_shared<ScalarType>(data_ptrs[0],
                                                                        m * n,
                                                                        copy));
            break;
        }
        case invlib::Format::SparseCsc : {
            data_ptrs_.push_back(invlib::array::make_shared<ScalarType>(data_ptrs[0],
                                                                        nnz,
                                                                        copy));
            index_ptrs_.push_back(invlib::array::make_shared<IndexType>(index_ptrs[0],
                                                                         nnz,
                                                                         copy));
            index_ptrs_.push_back(invlib::array::make_shared<IndexType>(index_ptrs[1],
                                                                         n + 1,
                                                                         copy));
            break;
        }
        case invlib::Format::SparseCsr : {
            data_ptrs_.push_back(invlib::array::make_shared<ScalarType>(data_ptrs[0],
                                                                        nnz,
                                                                        copy));
            index_ptrs_.push_back(invlib::array::make_shared<IndexType>(index_ptrs[0],
                                                                        nnz,
                                                                        copy));
            index_ptrs_.push_back(invlib::array::make_shared<IndexType>(index_ptrs[1],
                                                                        m + 1,
                                                                        copy));
            break;
        }
        }
        PythonMatrix * ptr = new PythonMatrix(
            invlib::PythonMatrix<ScalarType, IndexType>{m, n, nnz,
                                                                            index_ptrs_,
                                                                            data_ptrs_,
                                                                            format_});
        return ptr;
    }

    void destroy_matrix(void * A)
    {
        delete reinterpret_cast<PythonMatrix *>(A);
    }

    size_t matrix_cols(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
        return A_.cols();
    }

    size_t matrix_rows(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
        return A_.rows();
    }

    size_t matrix_non_zeros(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
        return A_.non_zeros();
    }

    unsigned matrix_format(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
        return static_cast<unsigned>(A_.get_format());
    }

    void * matrix_element_pointer(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
        return static_cast<void *>(A_.get_element_pointer());
    }

    void * matrix_index_pointer(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
        return static_cast<void *>(A_.get_index_pointer());
    }

    void * matrix_start_pointer(void *A) {
        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
        return static_cast<void *>(A_.get_start_pointer());
    }

    //
    // Arithmetic
    //

    void* matrix_vector_multiply(void *A,
                                 void *u)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
        auto & u_ = *reinterpret_cast<PythonVector*>(u);
        auto   v_ = new PythonVector(A_ * u_);

        return v_;
    }

    void* matrix_vector_transpose_multiply(void *A,
                                          void *u)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
        auto & u_ = *reinterpret_cast<PythonVector*>(u);
        auto   v_ = new PythonVector(transp(A_) * u_);
        return v_;
    }

    void* matrix_matrix_multiply(void *A,
                                 void *B)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
        auto & B_ = *reinterpret_cast<PythonMatrix*>(B);
        auto   C_ = new PythonMatrix(A_ * B_);
        return C_;
    }

    void* matrix_matrix_transpose_multiply(void *A,
                                           void *B)
    {
        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
        auto & B_ = *reinterpret_cast<PythonMatrix*>(B);
        auto   C_ = new PythonMatrix(transp(A_) * B_);
        return C_;
    }


////////////////////////////////////////////////////////////////////////////////
// CG Solver
////////////////////////////////////////////////////////////////////////////////

    using SettingsType = invlib::CGPythonSettings<PythonVector>;
    using SolverType   = invlib::ConjugateGradient<SettingsType>;
    using StartVectorFunction = void (void *, const void *);

    SolverType * create_solver(ScalarType tolerance,
                               size_t step_limit,
                               int verbosity) {
        auto & solver   = * new SolverType(tolerance, verbosity);
        auto & settings = solver.get_settings();
        settings.tolerance  = tolerance;
        settings.step_limit = step_limit;

        return & solver;
    }

    void destroy_solver(void *ptr) {
        delete reinterpret_cast<SolverType *>(ptr);
    }

    ScalarType solver_get_tolerance(void * ptr) {
        auto & solver = * reinterpret_cast<SolverType *>(ptr);
        auto & settings = solver.get_settings();
        return settings.tolerance;
    }

    void solver_set_tolerance(void * ptr,
                              ScalarType tolerance) {
        auto & solver   = * reinterpret_cast<SolverType *>(ptr);
        auto & settings = solver.get_settings();
        settings.tolerance  = tolerance;
    }

    size_t solver_get_step_limit(void * ptr) {
        auto & solver = * reinterpret_cast<SolverType *>(ptr);
        auto & settings = solver.get_settings();
        return settings.step_limit;
    }

    void solver_set_step_limit(void * ptr,
                               size_t step_limit) {
        auto & solver = * reinterpret_cast<SolverType *>(ptr);
        auto & settings = solver.get_settings();
        settings.step_limit = step_limit;
    }

    void solver_set_start_vector_ptr(void * ptr,
                                     StartVectorFunction *start_vector_ptr) {
        auto & solver = * reinterpret_cast<SolverType *>(ptr);
        auto & settings = solver.get_settings();
        settings.start_vector_ptr = start_vector_ptr;
    }

    void * solver_solve(void * solver,
                        void * A,
                        void * b) {

        auto & solver_ = * reinterpret_cast<SolverType *>(solver);
        auto & A_      = * reinterpret_cast<PythonMatrix *>(A);
        auto & b_      = * reinterpret_cast<PythonVector *>(b);

        auto c_ = new PythonVector(solver_.solve(A_, b_));
        return reinterpret_cast<PythonVector *>(c_);
    }

////////////////////////////////////////////////////////////////////////////////
// Forward Model
////////////////////////////////////////////////////////////////////////////////

    using ForwardModel = invlib::PythonForwardModel<PythonMatrix, PythonVector>;
    using JacobianFunctionPointer = typename ForwardModel::JacobianFunctionPointer;
    using EvaluateFunctionPointer = typename ForwardModel::EvaluateFunctionPointer;

    struct ForwardModelStruct {
        size_t m;
        size_t n;
        void * jacobian_ptr;
        void * evaluate_ptr;
    };

    void * forward_model_evaluate(ForwardModelStruct f,
                                  void * x) {
        std::cout << "evaluate :: " << f.m << " :: " << f.n << std::endl;
        std::cout << "evaluate :: " << f.jacobian_ptr << " :: " << f.evaluate_ptr << std::endl;
        std::cout << "evaluate :: " << x << " :: " << f.evaluate_ptr << std::endl;
        auto fm = ForwardModel(f.m, f.n,
                               reinterpret_cast<JacobianFunctionPointer>(f.jacobian_ptr),
                               reinterpret_cast<EvaluateFunctionPointer>(f.evaluate_ptr));
        return new PythonVector(fm.evaluate(* reinterpret_cast<PythonVector *>(x)));
    }

    void * forward_model_jacobian(ForwardModelStruct f,
                                  void * x) {
        auto fm = ForwardModel(f.m, f.n,
                               reinterpret_cast<JacobianFunctionPointer>(f.jacobian_ptr),
                               reinterpret_cast<EvaluateFunctionPointer>(f.evaluate_ptr));
        PythonVector y{};
        return new PythonMatrix(fm.jacobian(* reinterpret_cast<PythonVector *>(x), y));
    }

    struct OptimizerStruct {
        unsigned int   type;
        ScalarType   * parameters;
        void         * solver;
    };

////////////////////////////////////////////////////////////////////////////////
// OEM
////////////////////////////////////////////////////////////////////////////////

    using PrecmatType      = invlib::PrecisionMatrix<PythonMatrix>;
    using OEMType          = invlib::MAP<ForwardModel,
                                         PythonMatrix,
                                         PrecmatType,
                                         PrecmatType,
                                         PythonVector>;

    void * oem(ForwardModelStruct f,
               void * s_a_inv,
               void * s_e_inv,
               void * x_a,
               void * x_0,
               OptimizerStruct opt)
    {
        // PrecmatType SaInv_(*reinterpret_cast<PythonMatrix *>(SaInv));
        // PrecmatType SeInv_(*reinterpret_cast<PythonMatrix *>(SeInv));

        // size_t m = s_e_inv.rows();
        // size_t n = s_a_inv.rows();

        // fm = PythonForwardModel(m, n,
        //                         reinterpret_cast<JacobianFunction>(f.jacobian_ptr),
        //                         reinterpret_cast<EvaluateFunction>(f.evaluate_ptr));

        // return fm.jacobian(x_a, x_0);

    //     auto & x_a_ = *reinterpret_cast<PythonVector *>(x_a);
    //     auto & y_  = *reinterpret_cast<PythonVector *>(y);

    //     Linear forward_model(*reinterpret_cast<MklCsc*>(K));


    //     SolverType cg = SolverType(1e-6, 1);
    //     Minimizer  gn(1e-6, 1, cg);
    //     LinearMAP  oem(forward_model, x_a_, SaInv_, SeInv_);

    //     auto x_ = new PythonVector{};

    //     oem.compute(*x_, y_, gn, 1);

    //     return x_;
    // }

    // void * forward_model_linear(void * K,
    //                             void * x)
    // {
    //     std::cout << K << " // " << x << std::endl;
    //     auto & x_ = *reinterpret_cast<PythonVector *>(x);
    //     Linear forward_model(*reinterpret_cast<MklCsc*>(K));
    //     auto y_ = new PythonVector{forward_model.evaluate(x_)};
    //     return y_;
    // }

    // void * covmat_multiply(void * S,
    //                        void * x)
    // {
    //     Covmat S_(*reinterpret_cast<MklCsc*>(S));
    //     auto & x_ = *reinterpret_cast<PythonVector *>(x);
    //     auto y_ = new PythonVector(inv(S_) * x_);
    //     return y_;
    }
}
