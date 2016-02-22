#include "../utility.h"
#include "algebra/matrix_identity.h"
#include "algebra/lazy_evaluation.h"
#include "algebra/matrix_inverse.h"

template
<
typename Real,
template <typename> class Vector,
template <typename> class Matrix
>
class SphereFunction
{

public:

    SphereFunction( unsigned int n_ )
        : n(n_) {}

    Real criterion( const Vector<Real> &x,
                    const Vector<Real> &dx ) const
    {
        return dx.norm();
    }

    Real cost_function(const Vector<Real> &x) const
    {
        Real res = 0.0;

        for (unsigned int i = 0; i < x.rows(); i++)
        {
            res += 0.5*x(i)*x(i);
        }

        return res;
    }

    Vector<Real> gradient( const Vector<Real> &x ) const
    {
        Vector<Real> g(n);

        for (unsigned int i = 0; i < n; i++)
        {
            g(i) = x[i];
        }

        return g;
    }

    Matrix<Real> Jacobian( const Vector<Real> &x ) const
    {
        Matrix<Real> A(n, n);

        for (unsigned int i = 0; i < n; i++)
        {
            for (unsigned int j = 0; j < n; j++)
            {
                A(i,j) = 0.0;
            }
            A(i,i) = x(i);
        }

        return A;
    }

    MatrixIdentity<Real> Hessian( const Vector<Real> &x ) const
    {
        MatrixIdentity<Real> I = MatrixIdentity<Real>();
        return I;
    }


private:

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

    Real criterion( const Vector &x,
                    const Vector &dx ) const
    {
        return dx.norm();
    }

    Vector step( const Vector &x ) const
    {
        Vector J(n);
        Matrix H(n, n);


        for (unsigned int i = 0; i < n; i++)
        {
            J(i) = ((double) (i + 2)) * pow(x(i), i+1);
            H(i, i) = ((double) (i + 1) * (i + 2)) * pow(x(i), i);
        }

        Vector dx = inv(H) * J;
        return dx;
    }

private:

    unsigned int n;

};
