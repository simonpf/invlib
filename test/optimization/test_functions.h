#include "algebra.h"
#include <iostream>

template
<
typename Real,
typename Vector,
typename Matrix
>
class SphereFunction
{

public:

    SphereFunction( unsigned int n_ )
        : n(n_), m(n_) {}

    Vector evaluate(const Vector &x)
    {
        return gradient(x);
    }

    Real criterion( const Vector &x,
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

    Vector gradient( const Vector &x ) const
    {
        Vector g; g.resize(n);

        for (unsigned int i = 0; i < n; i++)
        {
            g(i) = x[i];
        }

        return g;
    }

    Matrix Jacobian( const Vector &x ) const
    {
        return Hessian(x);
    }

    Matrix Hessian(const Vector &x) const
    {
        Matrix H{}; H.resize(n,n);
        for (unsigned int i = 0; i < n; i++)
        {
            for (unsigned int j = 0; j < n; j++)
            {
                H(i, j) = 0.0;
            }
            H(i, i) = 1.0;
        }
        return H;
    }

    const unsigned int n,m;
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
        : n_(n) {}

    inline Real m() {return n_;}
    inline Real n() {return n_;}

    Real criterion( const Vector &x,
                    const Vector &dx ) const
    {
        return dx.norm();
    }

    Vector step( const Vector &x ) const
    {
        Vector J(); J.resize(n_);
        Matrix H(); H.resize(n_, n_);


        for (unsigned int i = 0; i < n_; i++)
        {
            J(i) = ((double) (i + 2)) * pow(x(i), i+1);
            H(i, i) = ((double) (i + 1) * (i + 2)) * pow(x(i), i);
        }

        Vector dx = inv(H) * J;
        return dx;
    }

private:

    unsigned int n_;

};
