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
    case Format::SparseHyb : {                  \
        auto ptr = get_as<SparseHyb>();         \
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
        case Format::SparseHyb : {                \
            auto ptr_a = a->get_as<SparseHyb>();  \
            return ptr_a->f(b);                   \
        }                                         \
        }\

#include <cassert>
#include <cmath>
#include <iterator>
#include <memory>
#include <iostream>

#include "invlib/interfaces/python/general.h"
#include "invlib/interfaces/python/python_mpi.h"
#include "invlib/interfaces/python/python_vector.h"
#include "invlib/dense/matrix_data.h"
#include "invlib/blas/blas_vector.h"
#include "invlib/blas/blas_matrix.h"
#include "invlib/mkl/mkl_sparse.h"
#include "invlib/mpi/mpi_matrix.h"
#include "invlib/mpi/mpi_vector.h"

namespace invlib
{
    using invlib::Representation;

    // --------------------  //
    //  Architecture enum    //
    // --------------------  //

    template<typename ScalarType, typename IndexType>
    struct architecture_trait<ScalarType, IndexType, Architecture::Cpu> {
        using RealType     = ScalarType;
        using Vector   = BlasVector<ScalarType>;

        using DenseData = MatrixData<ScalarType>;
        using Dense =     BlasMatrix<ScalarType, MatrixData>;

        using SparseCscData = SparseData<ScalarType, IndexType, Representation::CompressedColumns>;
        using SparseCsrData = SparseData<ScalarType, IndexType, Representation::CompressedRows>;
        using SparseHybData = SparseData<ScalarType, IndexType, Representation::Hybrid>;

        using SparseHyb = MklSparse<ScalarType, Representation::Hybrid>;
        using SparseCsr = MklSparse<ScalarType, Representation::CompressedRows>;
        using SparseCsc = MklSparse<ScalarType, Representation::CompressedColumns>;
    };


    // -------------------  //
    //    Format enum       //
    // -------------------  //


    template<typename Arch>
        struct format_trait<Arch, typename Arch::Dense> {
        static constexpr Format format = Format::Dense;
    };

    template<typename Arch>
        struct format_trait<Arch, typename Arch::SparseCsc> {
        static constexpr Format format = Format::SparseCsc;
    };

    template<typename Arch>
        struct format_trait<Arch, typename Arch::SparseCsr> {
        static constexpr Format format = Format::SparseCsr;
    };

    template<typename Arch>
        struct format_trait<Arch, typename Arch::SparseHyb> {
        static constexpr Format format = Format::SparseHyb;
    };

    // -------------------  //
    //    Python Matrix     //
    // -------------------  //

    template <
        typename ScalarType,
        typename IndexType = unsigned long,
        Architecture arch = Architecture::Cpu
    >
    class PythonMatrix {
        public:

        /*! The floating point type used to represent scalars. */
        using Arch = architecture_trait<ScalarType, IndexType, arch>;

        using RealType   = ScalarType;
        using VectorType = typename Arch::Vector;
        using ResultType = PythonMatrix;

        using DenseData     = typename Arch::DenseData;
        using SparseCscData = typename Arch::SparseCscData;
        using SparseCsrData = typename Arch::SparseCsrData;
        using SparseHybData = typename Arch::SparseHybData;

        using Dense     = typename Arch::Dense;
        using SparseCsc = typename Arch::SparseCsc;
        using SparseCsr = typename Arch::SparseCsr;
        using SparseHyb = typename Arch::SparseHyb;

        // ------------------------------- //
        //  Constructors and Destructors   //
        // ------------------------------- //

        PythonMatrix() = delete;

        template <
            typename T,
            Format f = format_trait<Arch, T>::format
        >
        PythonMatrix(T *t)
        : matrix_ptr(t), format(f) {
            // Nothing to do here.
        }

        PythonMatrix(size_t m,
                     size_t n,
                     size_t nnz,
                     std::vector<std::shared_ptr<IndexType[]>>  index_pointers,
                     std::vector<std::shared_ptr<IndexType[]>>  start_pointers,
                     std::vector<std::shared_ptr<ScalarType[]>> data_pointers,
                     Format format_)
        : format(format_)
        {
            switch (format) {
            case Format::Dense : {
                if (!(data_pointers.size() == 1)
                    && (index_pointers.size() == 0)
                    && (start_pointers.size() == 0)) {
                    throw std::runtime_error("Provided number of data pointers "
                                             "and index pointers does not match "
                                             "what was expected for constructing"
                                             " a matrix in dense format.");
                }
                auto data  = DenseData(m, n, data_pointers[0]);
                matrix_ptr = std::make_shared<Dense>(data);
                break;
            }
            case Format::SparseCsc : {
                if (!(data_pointers.size() == 1)
                    && (index_pointers.size() == 1)
                    && (start_pointers.size() == 1)) {
                    throw std::runtime_error("Provided number of data pointers "
                                             "and index pointers does not match "
                                             "what was expected for constructing"
                                             " a matrix in sparse CSC format.");
                }
                auto data  = SparseCscData(m, n, nnz,
                                           index_pointers[0],
                                           start_pointers[0],
                                           data_pointers[0]);
                matrix_ptr = std::make_shared<SparseCsc>(data);
                break;
            }
            case Format::SparseCsr : {
                if (!(data_pointers.size() == 1)
                    && (index_pointers.size() == 1)
                    && (start_pointers.size() == 1)) {
                    throw std::runtime_error("Provided number of data pointers "
                                             "and index pointers does not match "
                                             "what was expected for constructing"
                                             " a matrix in sparse CSC format.");
                }
                /*
                 * Arguments of SparseData constructor are in order rows before columns
                 * the arguments must be reversed here.
                 */
                auto data  = SparseCsrData(m, n, nnz,
                                           start_pointers[0],
                                           index_pointers[0],
                                           data_pointers[0]);
                matrix_ptr = std::make_shared<SparseCsr>(data);
                break;
            }
            case Format::SparseHyb : {
                if (!(data_pointers.size() == 2) && (index_pointers.size() == 4)) {
                    throw std::runtime_error("Provided number of data pointers "
                                             "and index pointers does not match "
                                             "what was expected for constructing"
                                             " a matrix in sparse Hybrid format.");
                }
                /*
                 * Arguments of SparseData constructor are in order rows before columns
                 * the arguments must be reversed here.
                 */
                auto data_csc  = SparseCscData(m, n, nnz,
                                               index_pointers[0],
                                               start_pointers[0],
                                               data_pointers[0]);
                auto data_csr  = SparseCsrData(m, n, nnz,
                                               start_pointers[1],
                                               index_pointers[1],
                                               data_pointers[1]);
                matrix_ptr = std::make_shared<SparseHyb>(data_csc, data_csr);
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
                return a.non_zeros();
            }
            case Format::SparseCsr : {
                auto & a = *get_as<SparseCsr>();
                return a.non_zeros();
            }
            case Format::SparseHyb : {
                auto & a = *get_as<SparseHyb>();
                return a.non_zeros();
            }
            }
        }

        std::vector<ScalarType *> get_element_pointers() {
            std::vector<ScalarType *> pointers{};

            switch (format) {
            case Format::Dense : {
                pointers.push_back(get_as<Dense>()->get_element_pointer());
                break;
            }
            case Format::SparseCsc : {
                pointers.push_back(get_as<SparseCsc>()->get_element_pointer());
                break;
            }
            case Format::SparseCsr : {
                pointers.push_back(get_as<SparseCsr>()->get_element_pointer());
                break;
            }
            case Format::SparseHyb : {
                pointers = get_as<SparseHyb>()->get_element_pointers();
                break;
            }
            }
            return pointers;
        }

        std::vector<IndexType *> get_index_pointers() {
            std::vector<IndexType *> pointers{};

            switch (format) {
            case Format::Dense : {
                break;
            }
            case Format::SparseCsc : {
                pointers.push_back(get_as<SparseCsc>()->get_index_pointer());
                break;
            }
            case Format::SparseCsr : {
                pointers.push_back(get_as<SparseCsr>()->get_index_pointer());
                break;
            }
            case Format::SparseHyb : {
                pointers = get_as<SparseHyb>()->get_index_pointers();
                break;
            }
            }
            return pointers;
        }

        std::vector<IndexType *> get_start_pointers() {
            std::vector<IndexType *> pointers{};

            switch (format) {
            case Format::Dense : {
                break;
            }
            case Format::SparseCsc : {
                pointers.push_back(get_as<SparseCsc>()->get_start_pointer());
                break;
            }
            case Format::SparseCsr : {
                pointers.push_back(get_as<SparseCsr>()->get_start_pointer());
                break;
            }
            case Format::SparseHyb : {
                pointers = get_as<SparseHyb>()->get_start_pointers();
                break;
            }
            }
            return pointers;
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
