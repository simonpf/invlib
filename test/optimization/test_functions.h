#include "../utility.h"
#include "algebra/lazy_evaluation.h"
#include "algebra/matrix_inverse.h"

template
<
typename Real,
typename Vector,
typename Matrix
>
class SphereFunction
{

public:

    SphereFunction( unsigned int n )
        : J(n,n), n(n)
    {
        set_identity(J);
    }

    Real criterion( const Vector &xi,
                    const Vector &dx ) const
    {
        return dx.norm();
    }

    Real cost_function(const Vector &x) const
    {
        Real res = 0.0;

        for (unsigned int i = 0; i < x.rows(); i++)
        {
            res += 0.5*x(i)*x(i);
        }

        return res;
    }

    Matrix Hessian( const Vector &xi ) const
    {
        Matrix H(n,n);
        H.setIdentity();
        return H;
    }

    Vector gradient( const Vector &xi ) const
    {
        Vector g(n);

        for (unsigned int i = 0; i < n; i++)
        {
            g(i) = xi[i];
        }

        return g;
    }

private:

    Matrix J;
    unsigned int n;

};

template
<
typename Real,
typename Matrix,
typename Vector
>
class SumOfPowers
{

public:

    SumOfPowers( unsigned int n )
        : n(n) {}

    Real criterion( const Vector &xi,
                    const Vector &dx ) const
    {
        return dx.norm();
    }

    Vector step( const Vector &xi ) const
    {
        Vector J(n);
        Matrix H(n, n);


        for (unsigned int i = 0; i < n; i++)
        {
            J(i) = ((double) (i + 2)) * pow(xi(i), i+1);
            H(i, i) = ((double) (i + 1) * (i + 2)) * pow(xi(i), i);
        }

        Vector dx = inv(H) * J;
        return dx;
    }

private:

    unsigned int n;

};
