#ifndef OPTIMIZATION_LEVENBERG_MARQUARDT
#define OPTIMIZATION_LEVENBERG_MARQUARDT

#include "algebra/identity.h"
#include <iostream>

template
<
typename Real,
template<typename> class Vector,
template<typename> class Matrix,
template<typename, typename, typename> class CostFunction,
template<typename, template<typename> class > class DampingMatrix
>
class LevenbergMarquardt
{

public:

    LevenbergMarquardt( const DampingMatrix<Real, Matrix> &D_ )
        : lambda(4.0), maximum(100.0), decrease(2.0), increase(3.0),
        threshold(1.0), D(D_)
    {}

    int step( Vector<Real>       &dx,
              const Vector<Real> &x,
              Real         &cost,
              const Vector<Real> &g,
              const Matrix<Real> &B,
              const CostFunction<Real, Vector<Real>, Matrix<Real>> &J )
    {
        bool found_step = false;
        while (!found_step)
        {
            Matrix<Real> C  = D * lambda + B;
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

    Real lambda, maximum, increase, decrease, threshold;

    // Positive definite matrix defining the trust region sphere r < ||Mx||.
    DampingMatrix<Real, Matrix > D;

};

#endif // OPTIMIZATION_LEVENBERG_MARQUARDT
