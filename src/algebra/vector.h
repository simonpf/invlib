/** \file algebra/vector.h
 *
 * \brief Contains Vector class template for symbolic computations on generic
 * vector types.
 *
 */

#ifndef ALGEBRA_VECTOR
#define ALGEBRA_VECTOR

#include "matrix_sum.h"
#include <utility>
#include <iostream>

namespace invlib
{

/**
 * \brief Wrapper class for symbolic computations involving vectors.
 *
 * The Vector class provides an abstraction layer to delay computations
 * involving vectors. The following delayed operations are provided:
 *
 * - Addition: operator+()
 * - Subtraction: operator-()
 *
 * Those operations return a proxy type representing the computation, that
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

    // ------------------- //
    //  Element Iterator   //
    // ------------------- //

    class ElementIterator;

    /*!
     *\return An element iterator object pointing to the first element
     *in the vector.
     */
    ElementIterator begin() {return ElementIterator(this);};

    /*!
     *\return An element iterator pointing to the end of the vector.
     */
    ElementIterator end()   {return ElementIterator();};

    struct LEFT_VECTOR_MULTIPLY_NOT_SUPPORTED
    {
        using VectorBase = Vector;
    };

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    /* template <typename T> */
    /* using Product = LEFT_VECTOR_MULTIPLY_NOT_SUPPORTED; */

    /*! The basic scalar type. */
    using RealType   = typename Base::RealType;
    /*! The basic vector type  */
    using VectorType = typename Base::VectorType;
    /*! The basic matrix type. */
    using MatrixType = typename Base::MatrixType;
    /*! The type of the result of the expression */
    using ResultType = Vector;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    /*! Default constructor. */
    Vector() : Base() = default;

    /*! Default copy constructor.
     *
     * Call the Base copy constructor, which should perform a deep copy of
     * the provided vector v.
     *
     * /param v The vector v to be copied from.
     */
    Vector(const Vector& v) = default;

    /*! Default move constructor.
     *
     * Call the Base move constructor, which should perform a deep copy of
     * the provided vector v.
     *
     * /param v The vector v to be moved from.
     */
    Vector(Vector&& v)      = default;

    /*! Default assignment operator.
     *
     * Call the Base assignment operator, which should copy the values of
     * the provided vector v into this object.
     *
     * \param v The vector v to be assigned from.
     */
    Vector& operator=(const Vector& v) = default;

    /*! Default move assignment operator.
     *
     * Call the Base move assignment operator, which should copy the values of
     * the provided vector v into this object.
     *
     * \param v The vector v to be moved from.
     */
    Vector& operator=(Vector&& v) = default;

    /*! Move base object into vector.
     *
     * Moves the provided base object into the vector. In this way results
     * from arithmetic operations on the base type can be moved directly
     * into a Vector object in order to avoid expensive copying.
     *
     * \param v The base vector to be moved from.
     */
    Vector(Base&& v);

    // --------------------- //
    // Arithmetic Operators  //
    // --------------------- //

    /*! Proxy type for the sum of two vectors. */
    template <typename T>
        using Sum = MatrixSum<const Vector &, T, MatrixBase>;

    /*! Create sum expression.
     *
     * \tparam T The type of the object to add to the vector.
     * \return The algebraic expression object representing the sum of this
     * this vector and the given argument.
     */
    template<typename T>
    Sum<T> operator+(T &&B) const;

    /*! Proxy type for the difference of two vectors. */
    template <typename T>
    using Difference = MatrixDifference<const Vector &, T, MatrixBase>;

    /*! Create difference expression.
     *
     * \tparam The type of the object to add to the vector.
     * \return The algebraic expression object representing the difference
     * of this vector and the given argument.
     */
    template <typename T>
	auto operator-(const T &C) const -> Difference<T>;

};

/*! Dot product of two vectors.
 *
 * The dot product is computed by callling the dot(T t) member function
 * of the first argument vector with the second argument vector are argument.
 *
 * \tparam Base The fundamental vector type.
 * \return The dot product of the two vectors.
 */
template<typename Base>
auto dot(const Vector<Base>& v, const Vector<Base>& w) -> decltype(v.dot(w));

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
        : v(v_), k(0), n(v_->rows()) {}

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

#include "vector.cpp"

}      // namespace invlib

#endif // ALGEBRA_VECTOR
