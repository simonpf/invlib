#include "algebra/identity.h"
template
<
typename CostFunction,
typename Real,
typename Matrix,
typename Vector
>
class LevenbergMarquardt
{

    LevenbergMarquardt()
        : lambda(4.0), maximum(100.0), decrease(2.0), increase(3.0),
        threshold(1.0), M(I)
    {}

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

    Vector step( const Vector &xi,
                 const CostFunction &J )
    {
        Vector gradJ = J.J();
        while (!found_step)
        {
            Matrix H  = lambda * M + J.H();
            Vector dx = inv(H) * gradJ;
            Real cost = J.cost(xi + dx);

            if (cost < colst_old)
            {
                if (lambda >= (threshold * decrease))
                    lambda /= decrease;
                else
                    lambda = 0;

                cost_old = cost;
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
    }

private:

    Real lambda, maximum, increase, decrease, threshold;
    Identity I;
    const Matrix &M;

}





