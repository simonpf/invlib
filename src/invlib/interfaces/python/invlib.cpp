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

using ScalarType = @PREC@;
using IndexType = MKL_INT;
constexpr invlib::Architecture arch = invlib::Architecture::@ARCH@;

using PythonMatrix = invlib::Matrix<invlib::PythonMatrix<ScalarType, IndexType, arch>>;
using PythonVector = invlib::Vector<typename PythonMatrix::VectorType>;

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
        auto *v = new PythonVector(invlib::VectorData<ScalarType>(n, data_));
        return v;
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

    struct MatrixStruct {
        size_t m, n, nnz;
        unsigned   format;
        ScalarType * data_pointers[2];
        IndexType *  index_pointers[2];
        IndexType *  start_pointers[2];
    };

    using DataPointer   = std::shared_ptr<ScalarType[]>;
    using DataPointers  = std::vector<DataPointer>;

    using IndexPointer  = std::shared_ptr<IndexType[]>;
    using IndexPointers = std::vector<IndexPointer>;

    using invlib::array::make_shared;

    void* create_matrix(MatrixStruct matrix_struct, bool copy)
    {
        DataPointers  data_pointers{};
        IndexPointers index_pointers{};
        IndexPointers start_pointers{};

        size_t m   = matrix_struct.m;
        size_t n   = matrix_struct.n;
        size_t nnz = matrix_struct.nnz;

        auto format = static_cast<invlib::Format>(matrix_struct.format);

        switch (format) {
        case invlib::Format::Dense : {
            auto ptr = make_shared<ScalarType>(matrix_struct.data_pointers[0], m * n, copy);
            data_pointers.push_back(ptr);
            break;
        }
        case invlib::Format::SparseCsc : {
            auto ptr = make_shared<ScalarType>(matrix_struct.data_pointers[0], nnz, copy);
            data_pointers.push_back(ptr);

            auto ind_ptr = make_shared<IndexType>(matrix_struct.index_pointers[0], nnz, copy);
            index_pointers.push_back(ind_ptr);

            ind_ptr = make_shared<IndexType>(matrix_struct.start_pointers[0], nnz, copy);
            start_pointers.push_back(ind_ptr);
            break;
        }
        case invlib::Format::SparseCsr : {
            auto ptr = make_shared<ScalarType>(matrix_struct.data_pointers[0], nnz, copy);
            data_pointers.push_back(ptr);

            auto ind_ptr = make_shared<IndexType>(matrix_struct.index_pointers[0], nnz, copy);
            index_pointers.push_back(ind_ptr);

            ind_ptr = make_shared<IndexType>(matrix_struct.start_pointers[0], nnz, copy);
            start_pointers.push_back(ind_ptr);
            break;
        }
        case invlib::Format::SparseHyb : {

            auto ptr = make_shared<ScalarType>(matrix_struct.data_pointers[0], nnz, copy);
            data_pointers.push_back(ptr);
            ptr = make_shared<ScalarType>(matrix_struct.data_pointers[1], nnz, copy);
            data_pointers.push_back(ptr);

            auto ind_ptr = make_shared<IndexType>(matrix_struct.index_pointers[0], nnz, copy);
            index_pointers.push_back(ind_ptr);
            ind_ptr = make_shared<IndexType>(matrix_struct.index_pointers[1], nnz, copy);
            index_pointers.push_back(ind_ptr);

            ind_ptr = make_shared<IndexType>(matrix_struct.start_pointers[0], nnz, copy);
            start_pointers.push_back(ind_ptr);
            ind_ptr = make_shared<IndexType>(matrix_struct.start_pointers[1], nnz, copy);
            start_pointers.push_back(ind_ptr);

            break;
        }
        }
        PythonMatrix * ptr = new PythonMatrix(
            invlib::PythonMatrix<ScalarType, IndexType, arch>{
                m, n, nnz, index_pointers, start_pointers, data_pointers, format
                    });
        return ptr;
    }

    void destroy_matrix(void * A)
    {
        delete reinterpret_cast<PythonMatrix *>(A);
    }

    MatrixStruct matrix_info(void *A) {
        auto & A_ = * reinterpret_cast<PythonMatrix *>(A);
        MatrixStruct matrix_struct;
        matrix_struct.m   = A_.rows();
        matrix_struct.n   = A_.cols();
        matrix_struct.nnz = A_.non_zeros();
        matrix_struct.format = static_cast<unsigned>(A_.get_format());

        auto element_pointers = A_.get_element_pointers();

        for (size_t i = 0; i < 2; ++i) {
            matrix_struct.data_pointers[i] = nullptr;
        }

        for (size_t i = 0; i < element_pointers.size(); ++i) {
            matrix_struct.data_pointers[i] = element_pointers[i];
        }

        auto index_pointers = A_.get_index_pointers();
        for (size_t i = 0; i < 2; ++i) {
            matrix_struct.index_pointers[i] = nullptr;
        }
        for (size_t i = 0; i < index_pointers.size(); ++i) {
            matrix_struct.index_pointers[i] = index_pointers[i];
        }

        auto start_pointers = A_.get_start_pointers();
        for (size_t i = 0; i < 2; ++i) {
            matrix_struct.start_pointers[i] = nullptr;
        }
        for (size_t i = 0; i < start_pointers.size(); ++i) {
            matrix_struct.start_pointers[i] = start_pointers[i];
        }

        return matrix_struct;
    }


//    size_t matrix_cols(void *A) {
//        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
//        return A_.cols();
//    }
//
//    size_t matrix_rows(void *A) {
//        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
//        return A_.rows();
//    }
//
//    size_t matrix_non_zeros(void *A) {
//        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
//        return A_.non_zeros();
//    }
//
//    unsigned matrix_format(void *A) {
//        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
//        return static_cast<unsigned>(A_.get_format());
//    }
//
//    void * matrix_element_pointer(void *A) {
//        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
//        return static_cast<void *>(A_.get_element_pointer());
//    }
//
//    void * matrix_index_pointer(void *A) {
//        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
//        return static_cast<void *>(A_.get_index_pointer());
//    }
//
//    void * matrix_start_pointer(void *A) {
//        auto & A_ = *reinterpret_cast<PythonMatrix*>(A);
//        return static_cast<void *>(A_.get_start_pointer());
//    }

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
        auto fm = ForwardModel(f.m, f.n,
                               reinterpret_cast<JacobianFunctionPointer>(f.jacobian_ptr),
                               reinterpret_cast<EvaluateFunctionPointer>(f.evaluate_ptr));
        return new PythonVector(fm.evaluate(* reinterpret_cast<PythonVector *>(x)));
    }

    void * forward_model_jacobian(ForwardModelStruct f,
                                  void * x,
                                  void * y) {
        auto fm = ForwardModel(f.m, f.n,
                               reinterpret_cast<JacobianFunctionPointer>(f.jacobian_ptr),
                               reinterpret_cast<EvaluateFunctionPointer>(f.evaluate_ptr));
        return new PythonMatrix(fm.Jacobian(* reinterpret_cast<PythonVector *>(x),
                                            * reinterpret_cast<PythonVector *>(y)));
    }

////////////////////////////////////////////////////////////////////////////////
// Optimizer
////////////////////////////////////////////////////////////////////////////////

    struct OptimizerStruct {
        unsigned int   type;
        ScalarType   * parameters;
    };

////////////////////////////////////////////////////////////////////////////////
// OEM
////////////////////////////////////////////////////////////////////////////////

    using PrecmatType      = invlib::PrecisionMatrix<PythonMatrix>;
    using MinimizerType    = invlib::GaussNewton<ScalarType, SolverType>;
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
               void * y,
               OptimizerStruct opt,
               SolverType * solver)
    {
        auto fm = ForwardModel(f.m, f.n,
                               reinterpret_cast<JacobianFunctionPointer>(f.jacobian_ptr),
                               reinterpret_cast<EvaluateFunctionPointer>(f.evaluate_ptr));

        PrecmatType s_a_inv_(*reinterpret_cast<PythonMatrix *>(s_a_inv));
        PrecmatType s_e_inv_(*reinterpret_cast<PythonMatrix *>(s_e_inv));


        auto & x_a_ = *reinterpret_cast<PythonVector *>(x_a);
        auto & y_   = *reinterpret_cast<PythonVector *>(y);

        auto x = new PythonVector();
        if (x_0) {
            *x = *reinterpret_cast<PythonVector *>(x_0);
        } else {
            x->resize(fm.n);
            *x = x_a_;
        }

        SolverType & cg = * reinterpret_cast<SolverType *>(solver);
        MinimizerType gn(1e-6, 1, cg);

        OEMType  oem(fm, x_a_, s_a_inv_, s_e_inv_);
        oem.compute(*x, y_, gn, 2);

        return x;
    }
}
