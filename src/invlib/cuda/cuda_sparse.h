/**
 * \file cuda/cuda_sparse.h
 *
 * \brief Contains the cuda sparse class which implements sparse matrix
 * type for use with the cuSparse library.
 *
 */
#ifndef CUDA_CUDA_SPARSE
#define CUDA_CUDA_SPARSE

#include "invlib/sparse/sparse_data.h"
#include "invlib/cuda/cusparse_generic.h"
#include "invlib/cuda/cuda_vector.h"

namespace invlib
{

/*! CuSparse Matrix descriptor struct. */
typedef struct {
    cusparseMatrixType_t MatrixType;
    cusparseFillMode_t   FillMode;
    cusparseDiagType_t   DiagType;
    cusparseIndexBase_t  IndexBase;
} cusparseMatDescr;

// ------------------------ //
// Cuda Sparse Matrix Class //
// ------------------------ //

/**
 * \brief Sparse Matrix Class for CUDA Architectures
 *
 * Template class for sparse matrices in compressed row and compressed column
 * formats on CUDA devices.
 *
 * Provides multiply(...) and multiply_transpose(...) member functions so that
 * they can be used as matrix type for the CG solver.
 *
 */
template<typename Real, Representation rep> class CudaSparse;

template<typename Real>
class CudaSparse<Real, Representation::CompressedColumns>
{
public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType   = Real;
    using VectorType = CudaVector<RealType>;
    using MatrixType = CudaSparse;
    using ResultType = CudaSparse;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    CudaSparse(const SparseData<Real, int, Representation::CompressedColumns> & base,
               CudaDevice & = Cuda::default_device);
    CudaSparse(const CudaSparse & )             = delete;
    CudaSparse(      CudaSparse &&)             = delete;
    CudaSparse & operator=(const CudaSparse & ) = delete;
    CudaSparse & operator=(      CudaSparse &&) = delete;

    size_t rows() const {return m;}
    size_t cols() const {return n;}

    // -----------//
    // Arithmetic //
    // ---------- //

    VectorType multiply(const VectorType &) const;
    VectorType transpose_multiply(const VectorType &) const;

protected:

    static cusparseMatDescr default_matrix_descriptor;

    size_t m, n, nnz;

    CudaDevice & device;

    std::shared_ptr<int *>  column_starts;
    std::shared_ptr<int *>  row_indices;
    std::shared_ptr<Real *> elements;

    CudaDeleter<int *>  int_deleter{};
    CudaDeleter<Real *> real_deleter{};
};

template<typename Real>
class CudaSparse<Real, Representation::CompressedRows>
{
public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType   = Real;
    using VectorType = CudaVector<RealType>;
    using MatrixType = CudaSparse;
    using ResultType = CudaSparse;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    CudaSparse(const SparseData<Real, int, Representation::CompressedRows> & base,
               CudaDevice & = Cuda::default_device);
    CudaSparse(const CudaSparse & )             = delete;
    CudaSparse(      CudaSparse &&)             = delete;
    CudaSparse & operator=(const CudaSparse & ) = delete;
    CudaSparse & operator=(      CudaSparse &&) = delete;

    size_t rows() const {return m;}
    size_t cols() const {return n;}

    // -----------//
    // Arithmetic //
    // ---------- //

    VectorType multiply(const VectorType &) const;
    VectorType transpose_multiply(const VectorType &) const;

protected:

    static cusparseMatDescr default_matrix_descriptor;

    size_t m, n, nnz;

    CudaDevice & device;

    std::shared_ptr<int *>  column_indices;
    std::shared_ptr<int *>  row_starts;
    std::shared_ptr<Real *> elements;

    CudaDeleter<int *>  int_deleter{};
    CudaDeleter<Real *> real_deleter{};
};

template<typename Real>
class CudaSparse<Real, Representation::Hybrid>
    : protected CudaSparse<Real, Representation::CompressedRows>,
      protected CudaSparse<Real, Representation::CompressedColumns>
{
public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType   = Real;
    using VectorType = CudaVector<RealType>;
    using MatrixType = CudaSparse;
    using ResultType = CudaSparse;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    CudaSparse(const SparseData<Real, int, Representation::Coordinates> & base,
               CudaDevice & = Cuda::default_device);
    CudaSparse(const CudaSparse & )             = delete;
    CudaSparse(      CudaSparse &&)             = delete;
    CudaSparse & operator=(const CudaSparse & ) = delete;
    CudaSparse & operator=(      CudaSparse &&) = delete;

    size_t rows() const {return m;}
    size_t cols() const {return n;}

    // -----------//
    // Arithmetic //
    // ---------- //

    VectorType multiply(const VectorType &) const;
    VectorType transpose_multiply(const VectorType &) const;

private:

    using CSRBase = CudaSparse<Real, Representation::CompressedRows>;
    using CSCBase = CudaSparse<Real, Representation::CompressedColumns>;

    // --------------//
    // Base Members  //
    // ------------- //

    using CSRBase::m;
    using CSRBase::n;
    using CSRBase::nnz;

    using CSRBase::default_matrix_descriptor;
    using CSRBase::device;
    using CSRBase::int_deleter;
    using CSCBase::real_deleter;

    using CSRBase::column_indices;
    using CSRBase::row_starts;

    using CSCBase::column_starts;
    using CSCBase::row_indices;
};

#include "cuda_sparse.cpp"

}      // namespace invlib
#endif // CUDA_CUDA_SPARSE
