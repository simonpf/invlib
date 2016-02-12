#ifndef ALGEBRA_IDENTITY
#define ALGEBRA_IDENTITY

#include <iostream>

template
<
typename Real,
template<typename> class Matrix
>
class IdentityMatrix
{

public:

    IdentityMatrix() : c(1.0) {}

    IdentityMatrix( Real c_ ) : c(c_) {}

    Matrix<Real> operator+(const Matrix<Real> &A)
    {
        assert(A.cols() == A.rows());

        Matrix<Real> B(A);

        for (unsigned int i = 0; i < A.cols(); i++)
        {
            B(i,i) += c;
        }
        return B;
    }

    Matrix<Real> operator*(Matrix<Real> &A)
    {
        assert(A.cols() == A.rows());

        Matrix<Real> B(A);
        return c*B;
    }

    IdentityMatrix operator*(Real c)
    {
        IdentityMatrix I(c);
        return I;
    }

private:

    Real c;
};

#endif //ALGEBRA_IDENTITY
