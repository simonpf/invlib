#ifndef OPTIMIZATION_LEVENBERG_MARQUARDT
#define OPTIMIZATION_LEVENBERG_MARQUARDT

#include "algebra/matrix_identity.h"
#include <iostream>

template
<
typename Real,
typename DampingMatrix
>
class LevenbergMarquardt
{

public:

    LevenbergMarquardt( const DampingMatrix &D_ )
        : tol(1e-5), max_iter(100), lambda(4.0), maximum(100.0), decrease(2.0),
        increase(3.0), threshold(1.0), D(D_)
    {}

    template
    <
        typename Vector,
        typename Matrix,
        typename CostFunction
    >
    int step( Vector             &dx,
              const Vector       &x,
              Real               &cost,
              const Vector       &g,
              const Matrix       &B,
              const CostFunction &J )
    {
        bool found_step = false;
        while (!found_step)
        {
            auto C  = D * lambda + B;
            dx = inv(C) * g;
            Real new_cost = J.cost_function(x + dx);

            if (new_cost < cost)
            {
                if (lambda >= (threshold * decrease))
                    lambda /= decrease;
                else
                    lambda = 0;

                cost = new_cost;
                found_step = true;
            }

            else
            {
                if (lambda < threshold)
                    lambda = threshold;
                else
                {
                    if (lambda < maximum)
                    {
                        lambda *= increase;
                        if (lambda > maximum)
                            lambda = maximum;
                    }
                    else
                    {
                        lambda = maximum + 1.0;
                        break;

                    }
                }
            }
        }
        return 0;
    }

protected:

    unsigned int & maximum_iterations()
    { return max_iter; }

    Real &tolerance()
    { return tol; }

    Real &lambda_start()
    { return lambda; }

    Real &lambda_maximum()
    { return maximum; }

    Real &lambda_decrease()
    { return decrease; }

    Real &lambda_increase()
    { return increase; }

    Real &lambda_threshold()
    { return threshold; }

private:

    Real tol, lambda, maximum, increase, decrease, threshold;
    unsigned int max_iter;

    // Positive definite matrix defining the trust region sphere r < ||Mx||.
    DampingMatrix D;

};

#endif // OPTIMIZATION_LEVENBERG_MARQUARDT
