/**
 * \file archtypes/sparse.h
 *
 * \brief Contains the Sparse class, which is an archetype for the sparse
 * matrices and used to convert between different representations and
 * library implementations.
 *
 */
#ifndef ARCHETYPES_SPARSE_ARCHETYPE
#define ARCHETYPES_SPARSE_ARCHETYPE

#include "invlib/archetypes/matrix_archetype.h"
#include "invlib/archetypes/vector_archetype.h"
#include "invlib/interfaces/eigen.h"

#include <algorithm>
#include <iterator>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace invlib
{

// ------------------------  //
//   Sparse Archetype Class  //
// ------------------------  //

enum class Representation {Coordinates, CompressedColumns, CompressedRows};

/*! Destructor for shared_ptr of array types.
 *
 * Calls the delete[] destructor for array types created using new[]. This
 * is necessary to properly destroy shared objects of array type.
 *
 * \tparam T The underlying type of the array T[].
 */
template <typename T>
struct ArrayDeleter
{
    ArrayDeleter()                      = default;
    ArrayDeleter(const ArrayDeleter & ) = default;
    ArrayDeleter(      ArrayDeleter &&) = default;

    void operator() (const T * ptr)
    {
        delete[] (* ptr);
        delete ptr;
    }
};

/*! Sparse archetype class template.
 *
 * Represents archetypes for different sparse matrix representations.
 *
 * \tparam Real The floating point type used to represent scalars.
 * \tparam Representation Representation type used for the sparse matrix.
 */
template
<
typename Real,
Representation rep
>
class SparseBase;

template
<
typename Real
>
class SparseBase<Real, Representation::Coordinates>
{
public:

    /*! The floating point type used to represent scalars. */
    using RealType   = Real;
    /*! The fundamental vector type used for the matrix algebra.*/
    using VectorType = VectorArchetype<Real>;
    /*! The fundamental matrix type used for the matrix algebra.*/
    using MatrixType = MatrixArchetype<Real>;
    /*! The result type of multiplying an algebraic expression with this
     * matrix from the right.
     */
    using ResultType = SparseBase;

    // ------------------- //
    //   Static Functions  //
    // ------------------- //

    static auto random(size_t m, size_t n) -> SparseBase;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    SparseBase(size_t m, size_t n);
    SparseBase(const std::vector<size_t> & row_indices,
                    const std::vector<size_t> & column_indices,
                    const std::vector<Real>   & elements);

    SparseBase(const SparseBase & )             = default;
    SparseBase(      SparseBase &&)             = default;
    SparseBase & operator=(const SparseBase & ) = default;
    SparseBase & operator=(      SparseBase &&) = default;

    SparseBase(const MatrixArchetype<Real> &);
    SparseBase(const SparseBase<Real, Representation::CompressedColumns> &);
    SparseBase(const SparseBase<Real, Representation::CompressedRows> &);

    ~SparseBase() = default;

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    void set(const std::vector<size_t> & row_indices,
             const std::vector<size_t> & column_indices,
             const std::vector<Real>   & elements);
    void resize(size_t i, size_t j);

    bool operator == (const SparseBase &) const;

    // --------------- //
    //   Data Access   //
    // --------------- //

    size_t * row_index_pointer()    {return * row_indices;}
    size_t * column_index_pointer() {return * column_indices;}
    Real   * element_pointer()      {return * elements;}

    const size_t * row_index_pointer()    const {return * row_indices;}
    const size_t * column_index_pointer() const {return * column_indices;}
    const Real   * element_pointer()      const {return * elements;}

    size_t rows()     const  {return m;}
    size_t cols()     const  {return n;}
    size_t non_zeros() const {return nnz;}

    // --------------- //
    //   Conversions   //
    // --------------- //

    operator SparseBase<Real, Representation::CompressedColumns>() const;
    operator SparseBase<Real, Representation::CompressedRows>() const;

    operator EigenSparse() const;

    operator MatrixArchetype<Real>() const;


private:

    size_t m, n, nnz;

    std::shared_ptr<size_t *> column_indices;
    std::shared_ptr<size_t *> row_indices;
    std::shared_ptr<Real *>   elements;

};

/*! Print sparse matrix to output stream. */
template <typename Real>
std::ostream & operator << (
    std::ostream &,
    const SparseBase<Real, Representation::Coordinates>&);

template
<
typename Real
>
class SparseBase<Real, Representation::CompressedColumns>
{
public:

    /*! The floating point type used to represent scalars. */
    using RealType   = Real;
    /*! The fundamental vector type used for the matrix algebra.*/
    using VectorType = VectorArchetype<Real>;
    /*! The fundamental matrix type used for the matrix algebra.*/
    using MatrixType = MatrixArchetype<Real>;
    /*! The result type of multiplying an algebraic expression with this
     * matrix from the right.
     */
    using ResultType = SparseBase;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    SparseBase(size_t m, size_t n, size_t nnz,
                    const std::shared_ptr<size_t *> & row_indices,
                    const std::shared_ptr<size_t *> & column_starts,
                    const std::shared_ptr<Real *>   & elements);

    SparseBase(const SparseBase & )             = default;
    SparseBase(      SparseBase &&)             = default;
    SparseBase & operator=(const SparseBase & ) = default;
    SparseBase & operator=(      SparseBase &&) = default;

    ~SparseBase() = default;

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    void resize(size_t i, size_t j);

    size_t * row_index_pointer()    {return * row_indices;}
    size_t * column_start_pointer() {return * column_starts;}
    Real   * element_pointer()      {return * elements;}

    const size_t * row_index_pointer()    const {return * row_indices;}
    const size_t * column_start_pointer() const {return * column_starts;}
    const Real   * element_pointer()      const {return * elements;}

    size_t rows()     const  {return m;}
    size_t cols()     const  {return n;}
    size_t non_zeros() const {return nnz;}

private:

    size_t m, n, nnz;

    std::shared_ptr<size_t *> row_indices;
    std::shared_ptr<size_t *> column_starts;
    std::shared_ptr<Real *>   elements;

};

/*! Print sparse matrix to output stream. */
template <typename Real>
std::ostream & operator << (
    std::ostream &,
    const SparseBase<Real, Representation::CompressedColumns>&);

template
<
typename Real
>
class SparseBase<Real, Representation::CompressedRows>
{
public:

    /*! The floating point type used to represent scalars. */
    using RealType   = Real;
    /*! The fundamental vector type used for the matrix algebra.*/
    using VectorType = VectorArchetype<Real>;
    /*! The fundamental matrix type used for the matrix algebra.*/
    using MatrixType = MatrixArchetype<Real>;
    /*! The result type of multiplying an algebraic expression with this
     * matrix from the right.
     */
    using ResultType = SparseBase;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    SparseBase() = delete;
    SparseBase(size_t m, size_t n, size_t nnz,
                    const std::shared_ptr<size_t *> & row_starts,
                    const std::shared_ptr<size_t *> & column_indices,
                    const std::shared_ptr<Real *>   & elements);

    SparseBase(const SparseBase & )             = default;
    SparseBase(      SparseBase &&)             = default;
    SparseBase & operator=(const SparseBase & ) = default;
    SparseBase & operator=(      SparseBase &&) = default;

    ~SparseBase() = default;

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    void resize(size_t i, size_t j);

    size_t * row_start_pointer()    {return * row_starts;}
    size_t * column_index_pointer() {return * column_indices;}
    Real   * element_pointer()      {return * elements;}

    const size_t * row_start_pointer()    const {return * row_starts;}
    const size_t * column_index_pointer() const {return * column_indices;}
    const Real   * element_pointer()      const {return * elements;}

    size_t rows()     const  {return m;}
    size_t cols()     const  {return n;}
    size_t non_zeros() const {return nnz;}

private:

    size_t m, n, nnz;

    std::shared_ptr<size_t *> row_starts;
    std::shared_ptr<size_t *> column_indices;
    std::shared_ptr<Real *>   elements;

};

/*! Print sparse matrix to output stream. */
template <typename Real>
std::ostream & operator << (
    std::ostream &,
    const SparseBase<Real, Representation::CompressedRows>&);

#include "sparse_base.cpp"

}      // namespace invlib
#endif // ARCHETYPES_SPARSE_ARCHETYPE
