/**
 * \file mkl/mkl_sparse.h
 *
 * \brief Spare matrix arithmetic using Intel MKL.
 *
 */
#ifndef MKL_MKL_SPARSE
#define MKL_MKL_SPARSE

#include <memory>

#include "invlib/blas/blas_vector.h"
#include "invlib/mkl/mkl_generic.h"
#include "invlib/sparse/sparse_data.h"

namespace invlib
{

// ----------------------- //
// MKL Sparse Matrix Class //
// ----------------------- //

/**
 * \brief MKL Sparse Matrix Arithmetic
 *
 * The MklSparse class template is derived from the SparseData class template and
 * extends it with matrix vector and transposed matrix vector multiplication
 * routines for use with the CG solver.
 *
 */
template<typename Real, Representation rep> class MklSparse;

template<typename Real> class MklSparse<Real, Representation::Coordinates>
    : public SparseData<Real, int, Representation::Coordinates>
{

public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType   = Real;
    using VectorType = BlasVector<RealType>;
    using MatrixType = BlasMatrix<RealType>;
    using ResultType = BlasMatrix<RealType>;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    MklSparse(const SparseData<Real, int, Representation::Coordinates> &);

    MklSparse()                               = delete;
    MklSparse(const MklSparse & )             = delete;
    MklSparse(      MklSparse &&)             = delete;
    MklSparse & operator=(const MklSparse & ) = delete;
    MklSparse & operator=(      MklSparse &&) = delete;

    // -----------//
    // Arithmetic //
    // ---------- //

    VectorType multiply(const VectorType &) const;
    VectorType transpose_multiply(const VectorType &) const;

private:

    // --------------//
    // Base Members  //
    // ------------- //

    using SparseData<Real, int, Representation::Coordinates>::nnz;
    using SparseData<Real, int, Representation::Coordinates>::m;
    using SparseData<Real, int, Representation::Coordinates>::n;

    using SparseData<Real, int, Representation::Coordinates>::row_indices;
    using SparseData<Real, int, Representation::Coordinates>::column_indices;
    using SparseData<Real, int, Representation::Coordinates>::elements;

};

template
<
typename Real,
Representation rep
>
class MklSparse
    : public SparseData<Real, int, rep>
{

public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType   = Real;
    using VectorType = BlasVector<RealType>;
    using MatrixType = BlasMatrix<RealType>;
    using ResultType = BlasMatrix<RealType>;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    MklSparse(const SparseData<Real, int, rep> &);

    MklSparse()                               = delete;
    MklSparse(const MklSparse & )             = delete;
    MklSparse(      MklSparse &&)             = delete;
    MklSparse & operator=(const MklSparse & ) = delete;
    MklSparse & operator=(      MklSparse &&) = delete;

    // -----------//
    // Arithmetic //
    // ---------- //

    VectorType multiply(const VectorType &) const;
    VectorType transpose_multiply(const VectorType &) const;

private:

    // --------------//
    // Base Members  //
    // ------------- //

    using SparseData<Real, int, rep>::nnz;
    using SparseData<Real, int, rep>::m;
    using SparseData<Real, int, rep>::n;
    using SparseData<Real, int, rep>::elements;

    using SparseData<Real, int, rep>::get_indices;
    using SparseData<Real, int, rep>::get_starts;

    // --------------//
    // Base Members  //
    // ------------- //

    std::shared_ptr<int *> ends;

};

template<typename Real> class MklSparse<Real, Representation::Hybrid>
    : protected SparseData<Real, int, Representation::CompressedRows>,
      protected SparseData<Real, int, Representation::CompressedColumns>
{

public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType   = Real;
    using VectorType = BlasVector<RealType>;
    using MatrixType = BlasMatrix<RealType>;
    using ResultType = BlasMatrix<RealType>;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    MklSparse(const SparseData<Real, int, Representation::Coordinates> &);

    MklSparse()                               = delete;
    MklSparse(const MklSparse & )             = delete;
    MklSparse(      MklSparse &&)             = delete;
    MklSparse & operator=(const MklSparse & ) = delete;
    MklSparse & operator=(      MklSparse &&) = delete;

    // -----------//
    // Arithmetic //
    // ---------- //

    VectorType multiply(const VectorType &) const;
    VectorType transpose_multiply(const VectorType &) const;

private:

    using CSRBase = SparseData<Real, int, Representation::CompressedRows>;
    using CSCBase = SparseData<Real, int, Representation::CompressedColumns>;

    // --------------//
    // Base Members  //
    // ------------- //

    using CSRBase::m;
    using CSRBase::n;
    using CSRBase::nnz;

    using CSRBase::column_indices;
    using CSRBase::row_starts;

    using CSCBase::column_starts;
    using CSCBase::row_indices;

};

#include "mkl_sparse.cpp"

}      // namespace invlib
#endif // MKL_MKL_SPARSE_H
