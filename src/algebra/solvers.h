#ifndef ALGEBRA_SOLVERS
#define ALGEBRA_SOLVERS

#include "algebra.h"
#include <iostream>

namespace invlib
{

/** file solvers.h
 * \brief Solver for linear systems.
 *
 * This file contains class that provide solvers for linear systems of the form
 *
 * \f[
 *  A x = b
 * \f]
 *
 * where \f$ A\f$ an full rank, square matrix \f$ A \in \mathbb{R}^{n \times n}\f$
 * and \f$ b \in \mathbb{R}^n \f$ a vector.
 *
 */

/** \brief Standard solver forwarding to underlying member functions.
 *
 * The Standard solver simply forwards the call to solve() to the underlying
 * member function Matrix::solve(const Vector&).
 */
class Standard
{
public:
    template
    <
    typename Vector,
    typename Matrix
    >
    Vector solve(const Matrix&A, const Vector& v)
    {
        return A.solve(v);
    }
};

/** \brief Conjugate gradient solver.
 *
 * The conjugate gradient solver computes an iterative solution of the system
 * using the conjugate gradient method. The convergence criterion used is the
 * Euclidean norm of the residual.
 *
 */
class ConjugateGradient
{

public:

    ConjugateGradient(double tol)
        : tolerance(tol) {}

    template
    <
    typename Vector,
    typename Matrix
    >
    Vector solve(const Matrix&A, const Vector& v)
    {

        using RealType = typename Vector::RealType;

        unsigned int n = v.rows();
        RealType tol, alpha, beta, rnorm;
        Vector x, r, p, xnew, rnew, pnew;

        x = v;
        r = A * x - v;
        p = -1.0 * r;

        int i = 0;
        while (r.norm() > tolerance)
        {
            alpha = dot(r, r) / dot(p, A * p);
            xnew  = x + alpha *     p;
            rnew  = r + alpha * A * p;
            beta  = dot(rnew, rnew) / dot(r, r);
            pnew  = beta * p - rnew;

            x = xnew;
            r = rnew;
            p = pnew;
        }

        return x;
    }

private:
    double tolerance;
};

}

#endif // ALGEBRA_SOLVERS

