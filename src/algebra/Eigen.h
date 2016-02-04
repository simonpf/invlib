#include "Eigen/Dense"
#include "lazy_evaluation.h"

template< typename Real >
class Vector : public Eigen::Matrix< Real, Eigen::Dynamic, 1 >
{ };

template< typename Real >
class Matrix : public Eigen::Matrix< Real, Eigen::Dynamic, Eigen::Dynamic >
{
public:

    typedef Eigen::Matrix< Real, Eigen::Dynamic, Eigen::Dynamic > Base;
    typedef LazyProduct< Matrix, Matrix, Vector<Real> > Product;

    using Base::Base;

    Matrix()
        : Base()
    { printf("Constructing object without argumens\n"); }

    Matrix( int m, int n )
        : Base(m,n)
    {
        printf("Constructing Object.\n");
    };

    Matrix( const Matrix& right )
        : Base()
    {
        printf("Copy constructor.\n");
    };

    ~Matrix()
    {
        printf("Destructing Object.\n");
    };

    Product operator*( Matrix &B )
    {
        return Product( *this, B );
    }

    Matrix delayed_multiplication( Matrix &B )
    {
        return Base::operator*(B);
    }

private:
    Eigen::Matrix< Real, Eigen::Dynamic, Eigen::Dynamic > M;
};
