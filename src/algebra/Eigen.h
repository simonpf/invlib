#include "Eigen/Dense"

#include "lazy_evaluation.h"
#include "matrix_inverse.h"

template< typename Real >
class Vector : public Eigen::Matrix< Real, Eigen::Dynamic, 1 >
{
    typedef Eigen::Matrix< Real, Eigen::Dynamic, 1 > Base;
    using Base::Base;
};

template< typename Real >
class Matrix : public Eigen::Matrix< Real, Eigen::Dynamic, Eigen::Dynamic >
{
public:

    typedef Eigen::Matrix< Real, Eigen::Dynamic, Eigen::Dynamic > Base;
    typedef LazyProduct< Matrix, Matrix, Vector<Real> > Product;

    using Base::Base;

    Product operator*( Matrix &B )
    {
        return Product( *this, B );
    }

    Vector<Real> operator*( Vector<Real> &v )
    {
        return Base::operator*(v);
    }

    Matrix delayed_multiplication( Matrix &B )
    {
        return Base::operator*(B);
    }

    Matrix invert() const
    {
        return this->inverse();
    }

    Vector<Real> solve(const Vector<Real> &v) const
    {
        return this->solve();
    }

private:
    Eigen::Matrix< Real, Eigen::Dynamic, Eigen::Dynamic > M;
};
