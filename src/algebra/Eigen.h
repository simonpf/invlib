
#include "Eigen/Dense"

#include "matrix_inverse.h"
#include "matrix_identity.h"
#include "matrix_zero.h"
#include "matrix.h"

class EigenWrapper : public Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
{
public:

    using Base = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    template <typename ...Args>
    EigenWrapper( Args&... params)
        : Base(std::forward<Args>(params)...) {}

    EigenWrapper(const Base& B)
        : Base(B) {}


    template <typename T>
    EigenWrapper transpose_multiply(const T& B) const
    {
        EigenWrapper C = this->transpose() * B;
        return C;
    }

    template <typename T>
    EigenWrapper transpose_add(const T& B) const
    {
        EigenWrapper C = this->transpose() + B;
        return C;
    }

    EigenWrapper invert() const
    {
        return EigenWrapper(this->colPivHouseholderQr().inverse());
    }

    template <typename Vector>
    Vector solve(const Vector &v) const
    {
        return Vector(this->colPivHouseholderQr().solve(v));
    }
};

using EigenVector = Vector<Eigen::Matrix<double, Eigen::Dynamic, 1>>;
using EigenMatrix = Matrix<EigenWrapper, EigenVector>;
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
