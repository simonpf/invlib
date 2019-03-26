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
    : public SparseData<Real, MKL_INT, Representation::Coordinates>
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

    MklSparse(const SparseData<Real, MKL_INT, Representation::Coordinates> &);

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

    using SparseData<Real, MKL_INT, Representation::Coordinates>::get_row_index_pointer;
    using SparseData<Real, MKL_INT, Representation::Coordinates>::get_column_index_pointer;
    using SparseData<Real, MKL_INT, Representation::Coordinates>::get_element_pointer;

protected:

    // --------------//
    // Base Members  //
    // ------------- //

    using SparseData<Real, MKL_INT, Representation::Coordinates>::nnz;
    using SparseData<Real, MKL_INT, Representation::Coordinates>::m;
    using SparseData<Real, MKL_INT, Representation::Coordinates>::n;

    using SparseData<Real, MKL_INT, Representation::Coordinates>::row_indices;
    using SparseData<Real, MKL_INT, Representation::Coordinates>::column_indices;
    using SparseData<Real, MKL_INT, Representation::Coordinates>::elements;

};

template
<
typename Real,
Representation rep
>
class MklSparse
    : public SparseData<Real, MKL_INT, rep>
{

public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType   = Real;
    using VectorType = void*;
    using MatrixType = MklSparse;
    using ResultType = MklSparse;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    MklSparse(const SparseData<Real, MKL_INT, rep> &);

    MklSparse()                               = delete;
    MklSparse(const MklSparse & )             = delete;
    MklSparse(      MklSparse &&)             = delete;
    MklSparse & operator=(const MklSparse & ) = delete;
    MklSparse & operator=(      MklSparse &&) = delete;

    // -----------//
    // Arithmetic //
    // ---------- //

    template < typename T, typename TT = typename T::ResultType >
    TT multiply(const T &) const;

    template <typename T>
    auto transpose_multiply(const T &) const -> typename T::ResultType;

    using SparseData<Real, MKL_INT, rep>::get_index_pointer;
    using SparseData<Real, MKL_INT, rep>::get_start_pointer;
    using SparseData<Real, MKL_INT, rep>::get_element_pointer;

protected:

    sparse_matrix_t mkl_matrix;

    // --------------//
    // Base Members  //
    // ------------- //

    using SparseData<Real, MKL_INT, rep>::nnz;
    using SparseData<Real, MKL_INT, rep>::m;
    using SparseData<Real, MKL_INT, rep>::n;
    using SparseData<Real, MKL_INT, rep>::elements;

};

template<typename Real> class MklSparse<Real, Representation::Hybrid>
    : protected MklSparse<Real, Representation::CompressedRows>,
      protected MklSparse<Real, Representation::CompressedColumns>
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

    MklSparse(const SparseData<Real, MKL_INT, Representation::Coordinates> &);
    MklSparse(const SparseData<Real, MKL_INT, Representation::CompressedColumns> &,
              const SparseData<Real, MKL_INT, Representation::CompressedRows> &);

    MklSparse()                               = delete;
    MklSparse(const MklSparse & )             = delete;
    MklSparse(      MklSparse &&)             = delete;
    MklSparse & operator=(const MklSparse & ) = delete;
    MklSparse & operator=(      MklSparse &&) = delete;

    // -----------//
    // Arithmetic //
    // ---------- //

    VectorType multiply(const VectorType &v) const {
        return CSRBase::multiply(v);
    }

    VectorType transpose_multiply(const VectorType &v) const {
        return CSCBase::transpose_multiply(v);
    }

    size_t rows() const {return CSRBase::rows();}
    size_t cols() const {return CSRBase::cols();}

    std::vector<Real *> get_element_pointers() {
        return {CSCBase::get_element_pointer(),
                CSRBase::get_element_pointer()};
    }

    std::vector<MKL_INT *> get_start_pointers() {
        return {CSCBase::get_start_pointer(),
                CSRBase::get_start_pointer()};
    }

    std::vector<MKL_INT *> get_index_pointers() {
        return {CSCBase::get_index_pointer(),
                CSRBase::get_index_pointer()};
    }

    using CSRBase = MklSparse<Real, Representation::CompressedRows>;
    using CSCBase = MklSparse<Real, Representation::CompressedColumns>;

    using CSRBase::non_zeros;

private:

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
