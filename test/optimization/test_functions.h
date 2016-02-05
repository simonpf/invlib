#include <iostream>
#include "../utility.h"
#include "algebra/lazy_evaluation.h"
#include "algebra/matrix_inverse.h"

template
<
typename Real,
typename Matrix,
typename Vector
>
class SphereFunction
{

public:

    SphereFunction( unsigned int n )
        : J_(n,n), n_(n)
    {
        set_identity(J_);
    }

    Real criterion( const Vector &xi,
                    const Vector &dx ) const
    {
        return dx.norm();
    }

    Vector step( const Vector &xi ) const
    {
        Vector dx  = J_ * J_ * xi;
        return dx;
    }

private:

    Matrix J_;
    unsigned int n_;

};

template
<
typename Real,
typename Vector,
typename Matrix
>
class SumOfPowers
{

public:

    SumOfPowers( unsigned int n )
        : n_(n) {}

    Real criterion( Vector &xi,
                    Vector &dx )
    {
        return dx.norm();
    }

    Vector step( Vector &xi )
    {
        Vector J(n_);
        Matrix H(n_, n_);


        for (unsigned int i = 0; i < n_; i++)
        {
            J(n_) = ((double) (i + 2)) * pow(xi(i), i+1);
            H(n_, n_) = ((double) (i + 1) * (i + 2)) * pow(xi(i), i);
        }

        return inv(H) * J * xi;
    }

private:

    unsigned int n_;

};
