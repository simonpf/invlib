#ifndef FORWARD_MODELS_H
#define FORWARD_MODELS_H

/** file forward_models.h
 * \brief Forward model classes
 *
 * This file provides generic forward model classes that simplify the creation
 * of forward models for computing MAP estimators.
 */

namespace invlib {

template <typename MatrixType> class LinearModel
{
public:

    LinearModel(const MatrixType & K_)
        : K(K_), m(K_.rows()), n(K_.cols())
    {
        // Nothing to do here.
    }

    template<typename T>
    auto evaluate(const T &x) const -> typename T::ResultType
    {
        return K * x;
    }

    template<typename T>
    const MatrixType & Jacobian(const T &x, T &y)
    {
        y = K * x;
        return K;
    }

    const unsigned int m,n;

private:

    const MatrixType & K;

};

template <typename MatrixType, typename VectorType>
class LinearizedModel
{
public:

    LinearizedModel(const MatrixType & K_,
                    const VectorType &x_a_)
    : K(K_), x_a(x_a_), m(K_.rows()), n(K_.cols())
    {
        // Nothing to do here.
    }

    template<typename T>
        auto evaluate(const T &x) const -> typename T::ResultType
    {
        return K * (x - x_a);
    }

    template<typename T>
        const MatrixType & Jacobian(const T &x, T &y)
    {
        y = K * (x - x_a);
        return K;
    }

    const unsigned int m,n;

private:

    const MatrixType & K;
    const VectorType & x_a;

};

}      // namespace invlib
#endif // FORWARD_MODELS_H
