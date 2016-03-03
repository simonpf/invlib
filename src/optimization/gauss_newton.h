#ifndef OPTIMIZATION_GAUSS_NEWTON_H
#define OPTIMIZATION_GAUSS_NEWTON_H

#include <stdio.h>
#include "algebra/solvers.h"

template <typename T>
void foo( T );

template
<
typename Real,
typename CostFunction,
typename Vector
>
int gauss_newton( CostFunction &J,
                  const Vector &x0,
                  Vector       &xi,
                  unsigned int max_iter,
                  Real         tol )
{

    bool converged     = false;
    unsigned int iter  = 0;

    Vector dx;
    xi = x0;

    while (!converged && (iter < max_iter))
    {

        dx = J.step(xi);

        if (J.criterion(xi, dx) < tol)
            converged = true;

        xi -= dx;
        iter++;
    }
}

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
              const Vector &x,
              const Vector &g,
              const Matrix &B,
              const CostFunction &J)
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

#endif //OPTIMIZATION_GAUSS_NEWTON
