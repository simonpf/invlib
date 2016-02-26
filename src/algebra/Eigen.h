
#include "Eigen/Dense"

#include "matrix_inverse.h"
#include "matrix_identity.h"
#include "matrix_zero.h"
#include "matrix.h"

class EigenMatrixWrapper : public Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
{
public:

    using Real = double;
    using Base = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    template <typename ...Args>
    EigenMatrixWrapper( Args &&... params)
        : Base(std::forward<Args>(params)...) {}

    EigenMatrixWrapper(const Base& B)
        : Base(B) {}


    template <typename T>
    EigenMatrixWrapper transpose_multiply(const T& B) const
    {
        EigenMatrixWrapper C = this->transpose() * B;
        return C;
    }

    template <typename T>
    EigenMatrixWrapper transpose_add(const T& B) const
    {
        EigenMatrixWrapper C = this->transpose() + B;
        return C;
    }

    EigenMatrixWrapper invert() const
    {
        return EigenMatrixWrapper(this->colPivHouseholderQr().inverse());
    }

    template <typename Vector>
    Vector solve(const Vector &v) const
    {
        return Vector(this->colPivHouseholderQr().solve(v));
    }
};

class EigenVectorWrapper : public Eigen::Matrix<double, Eigen::Dynamic, 1>
{
public:

    using Real = double;
    using Base = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    template <typename... Args>
        EigenVectorWrapper(Args&&... t) : Base(std::forward<Args>(t)...) {}

    template <typename T>
    EigenMatrixWrapper operator*(T&& t)
    {
        return this->Base::operator*(std::forward<T>(t));
    }

};

using EigenVector = Vector<EigenVectorWrapper>;
using EigenMatrix = Matrix<EigenMatrixWrapper, EigenVector>;
using I           = MatrixIdentity<double, EigenMatrix>;
using Zero        = MatrixZero<EigenMatrix>;

/* EigenVector operator+(const EigenVector &u, const EigenVector &v) */
/* { */
/*     return u + v; */
/* } */

double dot(const EigenVector &v, const EigenVector &w)
{
    return v.dot(w);
}
