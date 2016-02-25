#ifndef ALGEBRA_VECTOR
#define ALGEBRA_VECTOR

#include "matrix_sum.h"

#include <utility>
#include <iostream>

template
<
typename Base
>
class Vector : public Base
{
public:

    struct LEFT_VECTOR_MULTIPLY_NOT_SUPPORTED
    {
        using VectorBase = Vector;
    };

    template <typename T>
    using Product = LEFT_VECTOR_MULTIPLY_NOT_SUPPORTED;

    using MatrixBase = LEFT_VECTOR_MULTIPLY_NOT_SUPPORTED;
    using VectorBase = Vector;

    Vector()
        : Base() {}

    // ----------- //
    //   Addition  //
    // ----------- //

    Vector add(const Vector &v) const
    {
        Vector w(this->Base::operator+(v));
        return w;
    }

    // ----------- //
    //   Scaling   //
    // ----------  //

    template <typename Real>
    Vector scale(Real c) const
    {
        Vector v(this->Base::operator*(c));
        return v;
    }

    // ------------------ //
    // Addition Operator  //
    // -----------------  //

    template <typename T>
        using Sum = MatrixSum<Vector, T, MatrixBase>;

    template<typename T>
    Sum<T> operator+(const T &B) const
    {

        return Sum<T>(*this, B);
    }

    Vector(const Vector& v) = default;
    Vector(Vector&& v)      = default;

    Vector& operator=(const Vector& v) = default;
    Vector& operator=(Vector&& v) = default;

    Vector(const Base& v)
        : Base(v) {}

    Vector(Base&& v)
        : Base(v) {}

};

template<typename Base>
auto dot(const Vector<Base>& v, const Vector<Base>& w) -> decltype(v.dot(w))
{
    return v.dot(w);
}

#endif // ALGEBRA_VECTOR
