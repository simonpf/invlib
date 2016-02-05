#ifndef ALGEBRA_LAZY_EVALUATION
#define ALGEBRA_LAZY_EVALUATION

#include <stdio.h>

template
<
typename P,
typename Matrix
>
class LazyProduct
{

public:

    typedef LazyProduct< P, Matrix> Product;
    typedef LazyProduct< Product, Matrix> NestedProduct;

    LazyProduct( const P &Op1, const Matrix &Op2 )
        : A(Op1), B(Op2) {}

    NestedProduct operator*( const Matrix &C ) const
    {
        return NestedProduct( *this, C );
    }

    template <typename Vector>
    Vector operator*( const Vector &v ) const
    {
        Vector tmp1, tmp2, *result;
        tmp2.resize( B.rows() );
        tmp2 = B * v;
        result = compute(tmp1, tmp2);
        return *result;
    }

    operator Matrix()
    {
        Matrix tmp1 = A;
        Matrix tmp2 = tmp1.delayed_multiplication(B);
        return tmp2;
    }

    template <typename Vector>
    Vector* compute( Vector &tmp1, Vector &tmp2 )
    {
        tmp1.resize( B.rows() );
        tmp1 = B * tmp2;
        return A.compute( tmp2, tmp1 );
    }

private:

    // Operand references.
    const P      &A;
    const Matrix &B;

};

template < typename Matrix >
class LazyProduct< Matrix, Matrix >
{

public:

    typedef LazyProduct< Matrix, Matrix > Product;
    typedef LazyProduct< Product, Matrix > NestedProduct;

    LazyProduct( const Matrix &Op1, const Matrix &Op2 )
        : A(Op1), B(Op2) {}

    NestedProduct operator*( const Matrix &C ) const
    {
        return NestedProduct( *this, C );
    }

    template <typename Vector>
    Vector operator*( const Vector &v ) const
    {
        Vector tmp1, tmp2;
        tmp1.resize( B.rows() );
        tmp1 = B * v;
        tmp2.resize( A.rows() );
        tmp2 = A * tmp1;
        return tmp2;
    }

    operator Matrix()
    {
        Matrix tmp = A.delayed_multiplication( B );
        return tmp;
    }

    template <typename Vector>
    Vector* compute( Vector &tmp1, Vector &tmp2 )
    {
        tmp1.resize( B.rows() );
        tmp1 = B * tmp2;
        tmp2.resize( A.rows() );
        tmp2 = A * tmp1;
        return &tmp2;
    }

private:

    // Operand references.
    const Matrix &A;
    const Matrix &B;

};

#endif //ALGEBRA_LAZY_EVALUATION
