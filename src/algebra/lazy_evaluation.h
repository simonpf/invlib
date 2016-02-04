#include <stdio.h>

template
<
typename P,
typename Matrix,
typename Vector
>
class LazyProduct
{

public:

    typedef LazyProduct< P, Matrix, Vector> Product;
    typedef LazyProduct< Product, Matrix, Vector> NestedProduct;

    LazyProduct( P &Op1, Matrix &Op2 )
        : A(Op1), B(Op2) {}

    NestedProduct operator*( Matrix &C )
    {
        return NestedProduct( *this, C );
    }

    Vector operator*( Vector &v )
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

    Vector* compute( Vector &tmp1, Vector &tmp2 )
    {
        tmp1.resize( B.rows() );
        tmp1 = B * tmp2;
        return A.compute( tmp2, tmp1 );
    }

private:

    // Operand references.
    P      &A;
    Matrix &B;

};

template
<
typename Matrix,
typename Vector
>
class LazyProduct< Matrix, Matrix, Vector >
{

public:

    typedef LazyProduct< Matrix, Matrix, Vector> Product;
    typedef LazyProduct< Product, Matrix, Vector> NestedProduct;

    LazyProduct( Matrix &Op1, Matrix &Op2 )
        : A(Op1), B(Op2) {}

    NestedProduct operator*( Matrix &C )
    {
        return NestedProduct( *this, C );
    }

    Vector operator*( Vector &v )
    {
        Vector tmp1, tmp2, &result;
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
    Matrix &A;
    Matrix &B;

};
