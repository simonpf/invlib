/**
 * \file interfaces/python/python_matrix.h
 *
 * \brief Matrix class providing an interface for Python dense
 * and sparse matrices.
 *
 */
#ifndef INTERFACES_PYTHON_PYTHON_MATRIX
#define INTERFACES_PYTHON_PYTHON_MATRIX

#define RESOLVE_FORMAT(f)                       \
    switch (this->format) {                     \
    case Format::Dense : {                      \
        auto ptr = get_as<Dense>();             \
        return ptr->f();                        \
    }                                           \
    case Format::SparseCsc : {                  \
        auto ptr = get_as<SparseCsc>();         \
        return ptr->f();                        \
    }                                           \
    case Format::SparseCsr : {                  \
        auto ptr = get_as<SparseCsr>();         \
        return ptr->f();                        \
    }                                           \
    }                                           \

#define RESOLVE_FORMAT2(f, a, b) \
        switch (this->format) {\
        case Format::Dense : {\
            auto ptr_a = a->get_as<Dense>();\
            return ptr_a->f(b);\
           }\
        case Format::SparseCsc : {\
            auto ptr_a = a->get_as<SparseCsc>();\
            return ptr_a->f(b);\
            }\
        case Format::SparseCsr : {\
            auto ptr_a = a->get_as<SparseCsr>();\
            return ptr_a->f(b);\
        }\
        }\

#include <cassert>
#include <cmath>
#include <iterator>
#include <memory>
#include <iostream>

#include "invlib/interfaces/python/python_vector.h"
#include "invlib/dense/matrix_data.h"
#include "invlib/blas/blas_vector.h"
#include "invlib/blas/blas_matrix.h"
#include "invlib/mkl/mkl_sparse.h"

namespace invlib
{
    using invlib::Representation;

    // -------------------  //
    //    Format enum       //
    // -------------------  //

    enum class Format : unsigned {Dense, SparseCsc, SparseCsr, SparseHyb};

    template<typename T> struct format_trait;

    template<typename T1, template <typename> typename T2>
    struct format_trait<BlasMatrix<T1, T2>> {
        static constexpr Format format = Format::Dense;
    };

    template<typename T1>
    struct format_trait<MklSparse<T1, Representation::CompressedColumns>> {
        static constexpr Format format = Format::SparseCsc;
    };

    template<typename T1>
    struct format_trait<MklSparse<T1, Representation::CompressedRows>> {
        static constexpr Format format = Format::SparseCsr;
    };

    template<typename T1>
        struct format_trait<MklSparse<T1, Representation::Hybrid>> {
        static constexpr Format format = Format::SparseHyb;
    };

    // -------------------  //
    //    Python Matrix     //
    // -------------------  //

    template <
        typename ScalarType,
        typename IndexType = unsigned long
    >
    class PythonMatrix {
        public:

        /*! The floating point type used to represent scalars. */
        using RealType     = ScalarType;
        using VectorType   = BlasVector<ScalarType>;
        using MatrixType   = PythonMatrix<ScalarType, IndexType>;
        using ResultType   = PythonMatrix<ScalarType, IndexType>;

        using DenseData = MatrixData<ScalarType>;
        using Dense =     BlasMatrix<ScalarType, MatrixData>;

        using SparseCscData = SparseData<ScalarType,
                                         IndexType,
                                         Representation::CompressedColumns>;
        using SparseCsc = MklSparse<ScalarType, Representation::CompressedColumns>;

        using SparseCsrData = SparseData<ScalarType,
                                         IndexType,
                                         Representation::CompressedRows>;
        using SparseCsr = MklSparse<ScalarType, Representation::CompressedRows>;

        using SparseHybData = SparseData<ScalarType,
                                         IndexType,
                                         Representation::Hybrid>;
        using SparseHyb = MklSparse<ScalarType,
                                    Representation::Hybrid>;

        // ------------------------------- //
        //  Constructors and Destructors   //
        // ------------------------------- //

        PythonMatrix() = delete;

        template <
            typename T,
            Format f = format_trait<T>::format
        >
        PythonMatrix(T *t)
        : matrix_ptr(t), format(f) {
            // Nothing to do here.
        }

        PythonMatrix(size_t m,
                     size_t n,
                     size_t nnz,
                     std::vector<std::shared_ptr<IndexType[]>> index_ptrs,
                     std::vector<std::shared_ptr<ScalarType[]>> data_ptrs,
                     Format format_)
        : format(format_)
        {
            switch (format) {
            case Format::Dense : {
                if (!(data_ptrs.size() == 1) && (index_ptrs.size() == 0)) {
                    throw std::runtime_error("Provided number of data pointers "
                                             "and index pointers does not match "
                                             "what was expected for constructing"
                                             " a matrix in dense format.");
                }
                auto data  = DenseData(m, n, data_ptrs[0]);
                matrix_ptr = std::make_shared<Dense>(data);
                break;
            }
            case Format::SparseCsc : {
                if (!(data_ptrs.size() == 1) && (index_ptrs.size() == 2)) {
                    throw std::runtime_error("Provided number of data pointers "
                                             "and index pointers does not match "
                                             "what was expected for constructing"
                                             " a matrix in sparse CSC format.");
                }
                auto data  = SparseCscData(m, n, nnz,
                                           index_ptrs[0],
                                           index_ptrs[1],
                                           data_ptrs[0]);
                matrix_ptr = std::make_shared<SparseCsc>(data);
                break;
            }
            case Format::SparseCsr : {
                if (!(data_ptrs.size() == 1) && (index_ptrs.size() == 2)) {
                    throw std::runtime_error("Provided number of data pointers "
                                             "and index pointers does not match "
                                             "what was expected for constructing"
                                             " a matrix in sparse CSC format.");
                }
                /*
                 * Arguments of SparseData constructor are order rows before columns
                 * the arguments must be reversed here.
                 */
                auto data  = SparseCsrData(m, n, nnz,
                                           index_ptrs[1],
                                           index_ptrs[0],
                                           data_ptrs[0]);
                matrix_ptr = std::make_shared<SparseCsr>(data);
                break;
            }
            }
        }

        PythonMatrix(const PythonMatrix &)             = default;
        PythonMatrix(PythonMatrix &&)                  = default;
        PythonMatrix& operator=(const PythonMatrix &)  = default;
        PythonMatrix& operator=(PythonMatrix &&)       = default;

        ~PythonMatrix() = default;

        size_t rows() {
            RESOLVE_FORMAT(rows);
        }

        size_t cols() {
            RESOLVE_FORMAT(cols);
        }

        IndexType non_zeros() {
            switch (format) {
            case Format::Dense : {
                auto & a = *get_as<Dense>();
                return a.rows() * a.cols();
            }
            case Format::SparseCsc : {
                auto & a = *get_as<SparseCsc>();
                a.non_zeros();
            }
            case Format::SparseCsr : {
                auto & a = *get_as<SparseCsr>();
                return a.non_zeros();
            }
            }
        }

        ScalarType * get_element_pointer() {
            RESOLVE_FORMAT(get_element_pointer);
        }


        IndexType * get_index_pointer() {
            switch (format) {
            case Format::Dense : {
                throw std::runtime_error("Matrix in dense format has no "
                                         "index pointer array.");
                break;
            }
            case Format::SparseCsc : {
                auto & a = *get_as<SparseCsc>();
                return a.get_index_pointer();
            }
            case Format::SparseCsr : {
                auto & a = *get_as<SparseCsr>();
                return a.get_index_pointer();
            }
            }
        }

        IndexType * get_start_pointer() {
            switch (format) {
            case Format::Dense : {
                throw std::runtime_error("Matrix in dense format has no "
                                         "start pointer array.");
                break;
            }
            case Format::SparseCsc : {
                auto a = get_as<SparseCsc>();
                return a->get_start_pointer();
            }
            case Format::SparseCsr : {
                auto a = get_as<SparseCsr>();
                return a->get_start_pointer();
            }
            }
        }

        Format get_format() const {
            return format;
        }

        template<typename T> T * get_as() {
            return reinterpret_cast<T *>(matrix_ptr.get());
        }

        template<typename T> const T * get_as() const {
            return reinterpret_cast<const T *>(matrix_ptr.get());
        }

    // ----------- //
    //  Arithmetic //
    // ----------- //

        void accumulate(const PythonMatrix &b) {
            if (!(format == Format::Dense && b.format == Format::Dense)) {
                throw std::runtime_error("Accumulate member function can only be"
                                         " applied to dense matrices.");
            }
            auto ptr_a = get_as<Dense>();
            auto ptr_b = get_as<Dense>();
            ptr_a->accumulate(*ptr_b);
        }

        PythonMatrix multiply(const PythonMatrix &b) const {
            if (!(format == Format::Dense && b.format == Format::Dense)) {
                throw std::runtime_error("Matrix-matrix multiplication can only be"
                                         "applied to dense matrices.");
            }
            auto ptr_a = get_as<Dense>();
            auto ptr_b = b.get_as<Dense>();

            auto ptr_c = new Dense(ptr_a->multiply(*ptr_b));
            return PythonMatrix(ptr_c);
        }

        VectorType multiply(const VectorType &b) const {
            std::cout << "mult?" << std::endl;
            RESOLVE_FORMAT2(multiply, this, b);
        }

        PythonMatrix transpose_multiply(const PythonMatrix &b) const {
            if (!(format == Format::Dense && b.format == Format::Dense)) {
                throw std::runtime_error("Matrix-matrix multiplication can only be"
                                         "applied to dense matrices.");
            }
            auto ptr_a = get_as<Dense>();
            auto ptr_b = b.get_as<Dense>();

            auto ptr_c = new Dense(ptr_a->transpose_multiply(*ptr_b));
            return PythonMatrix(ptr_c);
        }

        VectorType transpose_multiply(const VectorType &b) const {
            RESOLVE_FORMAT2(transpose_multiply, this, b);
        }

        private:

        std::shared_ptr<void> matrix_ptr;
        Format format;

        };

}      // namespace invlib
#endif // INTERFACES_PYTHON_PYTHON_MATRIX
