/**
 * @file interfaces/python/python_forward_model.h
 * @author Simon Pfreundschuh
 * @date 2019-02-22
 * @brief Forward model wrapper for Python functions.
 */
#ifndef INTERFACES_PYTHON_PYTHON_FORWARD_MODEL
#define INTERFACES_PYTHON_PYTHON_FORWARD_MODEL
#include <cstdint>

namespace invlib
{
template
<
    typename JacobianType,
    typename VectorType
>
class PythonForwardModel
{
public:

    using JacobianFunctionPointer = JacobianType * (*)(const VectorType * x,
                                               VectorType * y);
    using EvaluateFunctionPointer = VectorType * (*)(const VectorType * x);

    PythonForwardModel(size_t m_,
                       size_t n_,
                       JacobianFunctionPointer f_jacobian_,
                       EvaluateFunctionPointer f_evaluate_)
        : m(m_), n(n_), f_jacobian(f_jacobian_), f_evaluate(f_evaluate_)
    {
        // Nothing to do here.
    }

    const VectorType evaluate(const VectorType &x) {
        auto y = f_evaluate(&x);
        return *y;
    }

    const JacobianType Jacobian(const VectorType &x, VectorType &y) {
        if (y.rows() != m) {
            y.resize(m);
        }
        auto *K = f_jacobian(&x, &y);
        return *K;
    }

    const unsigned m, n;

private:

    JacobianFunctionPointer f_jacobian;
    EvaluateFunctionPointer f_evaluate;

};

}

#endif
