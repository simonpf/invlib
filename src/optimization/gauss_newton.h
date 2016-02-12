#ifndef OPTIMIZATION_GAUSS_NEWTON
#define OPTIMIZATION_GAUSS_NEWTON

#include <stdio.h>

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
typename Vector,
typename Matrix
>
class GaussNewton
{

public:

    template<typename T>
    int step( Vector       &dx,
              const Vector &x,
              Real         &cost,
              const Vector &g,
              const Matrix &B )
    {
        dx = inv(B) * g;
        return 0;
    }
};

#endif //OPTIMIZATION_GAUSS_NEWTON
