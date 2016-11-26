/**
 * \file sparse/sparse_base.h
 *
 * \brief Contains the SparseData class, which is a base class for the sparse
 * matrices and used as a base for different library implementations and to
 *  convert between different representations.
 *
 */
#ifndef SPARSE_SPARSE_BASE
#define SPARSE_SPARSE_BASE

#include "invlib/archetypes/matrix_archetype.h"
#include "invlib/archetypes/vector_archetype.h"
#include "invlib/utility/array_deleter.h"

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
Representation rep = Representation::Coordinates
>
class SparseData;

template
<
typename Real
>
class SparseData<Real, Representation::Coordinates>
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
    using ResultType = SparseData;

    // ------------------- //
    //   Static Functions  //
    // ------------------- //

    static auto random(size_t m, size_t n) -> SparseData;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    SparseData(size_t m, size_t n);
    SparseData(const std::vector<size_t> & row_indices,
                    const std::vector<size_t> & column_indices,
                    const std::vector<Real>   & elements);

    SparseData(const SparseData & )             = default;
    SparseData(      SparseData &&)             = default;
    SparseData & operator=(const SparseData & ) = default;
    SparseData & operator=(      SparseData &&) = default;

    SparseData(const MatrixArchetype<Real> &);
    SparseData(const SparseData<Real, Representation::CompressedColumns> &);
    SparseData(const SparseData<Real, Representation::CompressedRows> &);

    ~SparseData() = default;

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    void set(const std::vector<size_t> & row_indices,
             const std::vector<size_t> & column_indices,
             const std::vector<Real>   & elements);
    void resize(size_t i, size_t j);

    bool operator == (const SparseData &) const;

    // --------------- //
    //   Data Access   //
    // --------------- //

    size_t * get_row_index_pointer()    {return * row_indices;}
    size_t * get_column_index_pointer() {return * column_indices;}
    Real   * get_element_pointer()      {return * elements;}

    const size_t * get_row_index_pointer()    const {return * row_indices;}
    const size_t * get_column_index_pointer() const {return * column_indices;}
    const Real   * get_element_pointer()      const {return * elements;}

    size_t rows()     const  {return m;}
    size_t cols()     const  {return n;}
    size_t non_zeros() const {return nnz;}

    // --------------- //
    //   Conversions   //
    // --------------- //

    operator SparseData<Real, Representation::CompressedColumns>() const;
    operator SparseData<Real, Representation::CompressedRows>() const;

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
    const SparseData<Real, Representation::Coordinates>&);

template
<
typename Real
>
class SparseData<Real, Representation::CompressedColumns>
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
    using ResultType = SparseData;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    SparseData(size_t m, size_t n, size_t nnz,
                    const std::shared_ptr<size_t *> & row_indices,
                    const std::shared_ptr<size_t *> & column_starts,
                    const std::shared_ptr<Real *>   & elements);

    SparseData(const SparseData & )             = default;
    SparseData(      SparseData &&)             = default;
    SparseData & operator=(const SparseData & ) = default;
    SparseData & operator=(      SparseData &&) = default;

    ~SparseData() = default;

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    void resize(size_t i, size_t j);

    size_t * get_row_index_pointer()    {return * row_indices;}
    size_t * get_column_start_pointer() {return * column_starts;}
    Real   * get_element_pointer()      {return * elements;}

    const size_t * get_row_index_pointer()    const {return * row_indices;}
    const size_t * get_column_start_pointer() const {return * column_starts;}
    const Real   * get_element_pointer()      const {return * elements;}

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
    const SparseData<Real, Representation::CompressedColumns>&);

template
<
typename Real
>
class SparseData<Real, Representation::CompressedRows>
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
    using ResultType = SparseData;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    SparseData() = delete;
    SparseData(size_t m, size_t n, size_t nnz,
                    const std::shared_ptr<size_t *> & row_starts,
                    const std::shared_ptr<size_t *> & column_indices,
                    const std::shared_ptr<Real *>   & elements);

    SparseData(const SparseData & )             = default;
    SparseData(      SparseData &&)             = default;
    SparseData & operator=(const SparseData & ) = default;
    SparseData & operator=(      SparseData &&) = default;

    ~SparseData() = default;

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    void resize(size_t i, size_t j);

    size_t * get_row_start_pointer()    {return * row_starts;}
    size_t * get_column_index_pointer() {return * column_indices;}
    Real   * get_element_pointer()      {return * elements;}

    const size_t * get_row_start_pointer()    const {return * row_starts;}
    const size_t * get_column_index_pointer() const {return * column_indices;}
    const Real   * get_element_pointer()      const {return * elements;}

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
    const SparseData<Real, Representation::CompressedRows>&);

#include "sparse_data.cpp"

}      // namespace invlib
#endif // SPARSE_SPARSE_BASE
