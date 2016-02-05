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
    typedef LazyProduct< Matrix, Matrix> Product;

    using Base::Base;

    Product operator*( const Matrix &B ) const
    {
        return Product( *this, B );
    }

    Vector<Real> operator*( const Vector<Real> &v ) const
    {
        return Base::operator*(v);
    }

    Matrix delayed_multiplication( const Matrix &B ) const
    {
        return Base::operator*(B);
    }

    Matrix invert() const
    {
        return this->inverse();
    }

    Vector<Real> solve(const Vector<Real> &v) const
    {
        return this->solve(v);
    }

private:
    Eigen::Matrix< Real, Eigen::Dynamic, Eigen::Dynamic > M;
};
