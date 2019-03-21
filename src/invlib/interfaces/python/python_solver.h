/**
 * \file interfaces/python/python_solver.h
 *
 * \brief Interface for the conjugate gradient solver class.
 *
 */

#ifndef INTERFACES_PYTHON_PYTHON_SOLVER
#define INTERFACES_PYTHON_PYTHON_SOLVER

#include "invlib/algebra/solvers.h"

namespace invlib
{

template<typename VectorType>
struct CGPythonSettings {

    using RealType = typename VectorType::RealType;

    CGPythonSettings(double tol)
    : tolerance(tol), step_limit(1e6)
    {
        // Nothing to do here.
    }

    CGPythonSettings(double tol,
                     size_t step_lim)
        : tolerance(tol), step_limit(step_lim)
    {
        // Nothing to do here.
    }

    CGPythonSettings(const CGPythonSettings &)  = default;
    CGPythonSettings(      CGPythonSettings &&) = default;
    CGPythonSettings & operator=(const CGPythonSettings &)  = default;
    CGPythonSettings & operator=(      CGPythonSettings &&) = default;
    ~CGPythonSettings() = default;

    VectorType start_vector(const VectorType &w) {

        const void * w_ptr = reinterpret_cast<const void *>(&w);

        if (start_vector_ptr) {
            auto v = VectorType(w);
            start_vector_ptr(&v, w_ptr);
            return v;
        } else {
            return VectorType(0.0 * w);
        }
    }

    bool converged(const VectorType &r,
                   const VectorType &v) {
        ++steps;

        RealType t;
        if (relative) {
            t = r.norm() / v.norm();
        } else {
            t = r.norm();
        }
        if (t < tolerance) {
            return true;
        }

        if (steps >= step_limit) {
            return true;
        }
        return false;
    }

    void (*start_vector_ptr)(void *, const void *) = nullptr;
    bool     relative   = true;
    RealType tolerance  = 1e-6;
    size_t   steps = 0;
    size_t   step_limit = 1e3;

};

}      // namespace invlib
#endif // INTERFACES_PYTHON_PYTHON_SOLVER
