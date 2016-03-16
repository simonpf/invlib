/**
 * \file archtypes/dense_matrix.h
 *
 * \brief Contains the MatrixArchetype class, which is an archetype for the basic
 * matrix type used.
 *
 */
#ifndef ARCHETYPES_MATRIX_ARCHETYPE
#define ARCHETYPES_MATRIX_ARCHETYPE

#include "vector_archetype.h"
#include <memory>
#include <iterator>

// ------------------------  //
//   Matrix Archetype Class  //
// ------------------------  //

/*! Matrix archtype.
 * 
 * Implements a straight forward dense matrix class to verify the 
 * generic matrix algebra and illustrate the interface to the
 * fundamental matrix type.
 *
 * \tparam The floating point type used to represent scalars.
 */
template
<
typename Real
>
class MatrixArchetype
{
public:

    /*! The floating point type used to represent scalars. */
    using RealType   = double;
    /*! The fundamental vector type used for the matrix algebra.*/
    using VectorType = VectorArchetype<Real>;
    /*! The fundamental matrix type used for the matrix algebra.*/
    using MatrixType = MatrixArchetype<Real>;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    /*! Default constructor. */
    MatrixArchetype() = default;

    /*! Copy constructor.
     *
     * The copy constructor should implement a deep copy of the argument, i.e.
     * after the call the constructed object hold a matrix identical to the one
     * passed as argument, but that is completely independent of the argument
     */
    MatrixArchetype(const MatrixArchetype &);

    // Moves are not supported.
    MatrixArchetype(MatrixArchetype &&) = default;

    /*! Assignment operator.
     *
     * The assignment operator should also perform a deep copy of the argument
     * matrix, such that the assigned to object after the call is identical to
     * the provided argument but independent.
     */
    MatrixArchetype& operator=(const MatrixArchetype &);

    // Moves are not supports.
    MatrixArchetype& operator=(MatrixArchetype &&) = default;

    /*!
     * Frees all resources occupied by the matrix object.
     */
    ~MatrixArchetype() = default;

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    /*! Resize matrix.
     *
     * Resize the matrix to a \f$i \times j \f$ matrix.
     *
     * \param i Number of rows of the resized matrix.
     * \param j Number of columns of the resized matrix.
     */
    void resize(unsigned int i, unsigned int j);

    /*! Element access.
     * \param i The row of the element to access.
     * \param j The column of the element to access.
     * \return Lvalue reference to the matrix element in row i
     * and column j.
     */
    Real & operator()(unsigned int, unsigned int);

    /*! Constant element access.
     * \param i The row of the element to access.
     * \param j The column of the element to access.
     * \return The value of the matrix element in row i and column j.
     */
    Real   operator()(unsigned int, unsigned int) const;

    /*! Number of columns of the matrix.
     * \return The number of columns of the matrix.
     */
    unsigned int cols() const;

    /*! Number of rows of the matrix.
     * \return The number of rows of the matrix.
     */
    unsigned int rows() const;

    // ------------ //
    //  Arithmetic  //
    // ------------ //

    /*! Accumulate into matrix.
     *
     * Add the elements of the given matrix to this matrix object.
     *
     * \param The matrix B to acummulate into this matrix.
     */
    void accumulate(const MatrixArchetype &B);

    /*! Subtract from matrix.
     *
     * Subtract the elements of the given matrix from this matrix object.
     *
     * \param The matrix B to subtract from this matrix.
     */
    void subtract(const MatrixArchetype &);

    /*! Matrix-matrix product.
     *
     * Compute the matrix product \f$C = A B\f$ of this matrix \f$A\f$
     * and the given matrix \f$B\f$.
     *
     * \param B The right hand operator for the multiplication
     * \return The matrix containing the result \f$C\f$ of the matrix
     * matrix product.
     */
    MatrixArchetype multiply(const MatrixArchetype &) const;

    /*! Matrix-vector product.
     *
     * Compute the matrix product \f$w = A v\f$ of this matrix \f$A\f$
     * and the given vector \f$v\f$.
     *
     * \param v The vector to be multiplied from the right.
     * \return The matrix containing the result \f$w\f$ of the matrix-vector
     *  product.
     */
    template
    <
    typename Vector
    > 
    Vector multiply(const Vector &) const;

    /*! Scale matrix.
     *
     * \param c The scalar to scale the matrix with.
     */
    void scale(Real c);

private:

    unsigned int n,m;
    std::unique_ptr<Real[]> data;

};

#include "matrix_archetype.cpp"

#endif
