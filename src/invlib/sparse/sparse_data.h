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
#include "invlib/utility/functions.h"

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

/**
 * \briref Sparse Data Class Template
 *
 * Represents sparse matrix data in different representation.
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

/**
 * \brief Sparse Data in Coordinate Representation
 *
 * In coordinate representation a sparse matrix is represented by three arrays:
 *
 * - A row index array holding the row indices of the elements.
 * - A column index array holding the column indices of the elements.
 * - An element array holding the elements corresponding to the above indices.
 *
 * The arrays are sorted first with respect to row indices and then with respect
 * to column indices.
 *
 * A sparse data in coordinate representation can be constructed from three
 * vectors holding the row indices, column indices and element data. The vectors
 * will be sorted, but multiple entries will not be removed. The data can also
 * be read from a sparse matrix stored in Arts data format using read_matrix_arts()
 *
 * Sparse data in coordinate representation also serves as a general conversion
 * type between the different sparse representations.
 *
 * \tparam Real The floating point type used to represent scalars.
 * \tparam Representation Representation type used for the sparse matrix.
 */
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

/**
 * \brief Sparse Data in Compressed Column Representation
 *
 * In compressed column representation a sparse \p m-times-\p n matrix with
 * \p nnz entries is represented by three arrays:
 *
 * - A row index array of length \p nnz holding the row indices of the elements.
 * - A column start array of length \p n holding the start index of the
 *   elements of column i as the ith element.
 * - An element array of length \p nnz holding the elements corresponding
 *   to the above indices.
 *
 * The arrays are sorted first with respect to column indices and then with respect
 * to row indices.
 *
 * A sparse data in coordinate representation can be constructed from three
 * vectors holding the row indices, column starts and element data. The vectors
 * will be sorted, but multiple entries will not be removed.
 *
 * It is also possible to convert a sparse matrix in coordinate format into
 * a matrix in compressed  column format. This will require allocation of
 * an size_t array of length \p nnz, a Real array of length \p nnz and a
 * size_t array of length \p n as well as an additional temporary size_t array
 * for the indirect sorting of the arrays. The critical operation here is the
 * sorting which has a complexity of nnz * log(nnz).
 *
 * \tparam Real The floating point type used to represent scalars.
 * \tparam Representation Representation type used for the sparse matrix.
 */
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

/**
 * \brief Sparse Data in Compressed Row Representation
 *
 * In compressed row representation a sparse \p m-times-\p n matrix with
 * \p nnz entries is represented by three arrays:
 *
 * - A row start array of length \p m holding the index of the first
 *   element of row i as the ith element.
 * - A column index array of length \p nnz.
 * - An element array of length \p nnz holding the elements corresponding
 *   to the above indices.
 *
 * The arrays are sorted first with respect to row indices and then with respect
 * to column indices.
 *
 * A sparse data in coordinate representation can be constructed from three
 * vectors holding the row start indices, column indices and element data. The vectors
 * will be sorted, but multiple entries will not be removed.
 *
 * It is also possible to convert a sparse matrix in coordinate format into
 * a matrix in compressed row format. This will require allocation of
 * an size_t array of length \p m. No resorting is required.
 *
 * \tparam Real The floating point type used to represent scalars.
 * \tparam Representation Representation type used for the sparse matrix.
 */
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
