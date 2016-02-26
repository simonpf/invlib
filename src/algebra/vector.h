#ifndef ALGEBRA_VECTOR
#define ALGEBRA_VECTOR

#include "matrix_sum.h"

#include <utility>
#include <iostream>

/**
 * \brief Wrapper class for symbolic computations involving vectors.
 *
 * The Vector class provides an abstraction layer to delay computations
 * involving vectors. The following delayed operations are provided:
 *
 * - Addition: operator+()
 * - Subtraction: operator-()
 *
 * Those operations return a proxy type, representing the computation, that
 * can be combined with the other matrix operations. The computation is delayed
 * until the resulting proxy object is converted to a matrix or a vector.
 *
 * \tparam Base The undelying vector type to be used.
 *
 */
template
<
typename Base
>
class Vector : public Base
{
public:

    class ElementIterator;
    ElementIterator begin() {return ElementIterator(this);};
    ElementIterator end() {return ElementIterator();};

    struct LEFT_VECTOR_MULTIPLY_NOT_SUPPORTED
    {
        using VectorBase = Vector;
    };

    template <typename T>
    using Product = LEFT_VECTOR_MULTIPLY_NOT_SUPPORTED;

    using Real = typename Base::Real;
    using VectorBase = Vector;
    using MatrixBase = LEFT_VECTOR_MULTIPLY_NOT_SUPPORTED;

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

    void accum(const Vector &B)
    {
        this->operator+=(B);
    }

    void subtract(const Vector &B)
    {
        this->operator-=(B);
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

    template <typename T>
    using Difference = MatrixDifference<Vector, T, MatrixBase>;

    template <typename T>
    auto operator-(const T &C) const -> Difference<T> const
    {
        return Difference<T>(*this, C);
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

/**
 * \brief Iterator for element-wise acces.
 *
 * Iterates over the elements in the vector using
 * <tt>operator()(unsigned int)</tt> for element access. Assumes
 * the length of the vector can be obtained usin <tt>cols()</tt> and
 * that indexing starts at 0.
 *
 * \tparam Base The undelying vector type to be used.
 *
 */
template
<
typename Base
>
class Vector<Base>::ElementIterator
{
public:

    using VectorType = Vector<Base>;

    ElementIterator() {}

    ElementIterator(VectorType* v_)
        : v(v_), k(0), n(v_->cols()) {}

    const Real& operator*()
    {
        return v->operator()(k);
    }

    Real& operator++()
    {
        k++;
    }

    template <typename T>
    bool  operator!=(T dummy)
    {
        return !(k == n);
    }

private:

    VectorType *v;
    unsigned int k, n;
};

#endif // ALGEBRA_VECTOR
