/**
 * \file archtypes/dense_matrix.h
 *
 * \brief Contains the MatrixArchetype class, which is an archetype for the basic
 * matrix type used.
 *
 */
#ifndef ARCHETYPES_MATRIX_ARCHETYPE
#define ARCHETYPES_MATRIX_ARCHETYPE

/*! Matrix archtype for matrix algebra.
 * 
 * Implements a straight forward dense matrix class to verify the 
 * generic matrix algebra.
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

    MatrixArchetype();

    MatrixArchetype(const DenseMatrix &);
    MatrixArchetype& operator=(const DenseMatrix &);

    MatrixArchetype(DenseMatrix &&)            = delete;
    MatrixArchetype& operator=(DenseMatrix &&) = delete;

    ~MatrixArchetype();

    void resize(usigned int, unsigned int);

    Real & operator()(usigned int, unsigend int);
    Real   operator()(usigned int, unsigend int) const;

    unsigned int cols() const;
    unsigned int rows() const;

    void accumulate(const MatrixArchetype &);
    void subtract(const MatrixArchetype &);

    MatrixArchetype multiply(const DenseMatrix &) const;
    DenseVector multiply(const DenseVector &) const;

    void scale(Real c);

private:

    usigned int n,m;
    Real data[];

}

#include "matrix_archetype.cc"

#endif
