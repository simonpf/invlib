/** \file optimization/gauss_newton.h
 *
 * \brief Contains GaussNewton class template implementing the Gauss-Newton
 * optimization scheme.
 *
 */

#ifndef OPTIMIZATION_GAUSS_NEWTON_H
#define OPTIMIZATION_GAUSS_NEWTON_H

#include <stdio.h>
#include "invlib/algebra/solvers.h"

namespace invlib
{

/**
 * \brief Gauss-Newton method.
 *
 * Class template for a generic Gauss-Newton optimizer. Provides the function
 * step, which computes one step \f$ d\vec{x} \f$ of the Gauss-Newton method:
 *
 * \f[
 *    d\vec{x} &= -\mathbf{H}^{-1} \vec{g}
 * \f]
 *
 * where \f$ \mathbf{H} \f$ is the (approximate) Hessian of the cost function
 * and \f$ \vec{g} \f$ its gradient. The next iteration vector can then be
 * computed using \f$ \vec{x}_{i+1} = \vec{x}_{i} + d\vec{x} \f$.
 *
 * The method used for the solution of the subproblem
 *
 * \f[
 *    \mathbf{H}} d\vec{x} = -\vec{g}
 * \f]
 *
 * can be defined using the @Solver type. The default is to use the
 * the solve() member function of the given Hessian.
 *
 * \tparam Real The floating point type used to represent scalars.
 * \tparam Solver The Solver type to be used for the subproblem.
 */
template
<
typename Real,
typename Solver = Standard
>
class GaussNewton
{

public:

    GaussNewton(Solver solver_ = Standard())
    : tol(1e-5), max_iter(1000), solver(solver_) {}

    GaussNewton( Real tolerance,
                 unsigned int max_iter,
                 Solver solver_ = Standard() )
    : tol(tolerance), max_iter(max_iter), solver(solver_) {}

    template
    <
        typename Vector,
        typename Matrix,
        typename CostFunction
    >
    int step( Vector       &dx,
              const Vector &,
              const Vector &g,
              const Matrix &B,
              const CostFunction &)
    {
        dx = -1.0 * solver.solve(B, g);
        return 0;
    }

    unsigned int& maximum_iterations()
    { return max_iter; }

    void maximum_iterations(unsigned int n)
    { max_iter = n; }

    Real& tolerance()
    { return tol; }

private:

    Real tol;
    unsigned int max_iter;
    Solver solver;

};

}

#endif //OPTIMIZATION_GAUSS_NEWTON
