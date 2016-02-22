#ifndef OPTIMIZATION_MINIMIZE
#define OPTIMIZATION_MINIMIZE

template
<
typename Real,
typename Vector,
typename CostFunction,
typename Minimizer
>
int minimize( CostFunction &J,
              Minimizer &M,
              const Vector &x0,
              Vector       &xi,
              unsigned int max_iter,
              Real         tol )
{
    bool converged     = false;
    unsigned int iter  = 0;
    Real cost = J.cost_function(x0);

    Vector dx;
    xi = x0;

    while (!converged && (iter < max_iter))
    {

        auto g =  J.gradient(xi);
        auto H =  J.Hessian(xi);
        M.step( dx, xi, cost, g, H, J );

        if (J.criterion(xi, dx) < tol)
            converged = true;

        xi -= dx;
        iter++;
    }
    return 0;
}

#endif // OPTIMIZATION_MINIMIZE
