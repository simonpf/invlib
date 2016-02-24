#ifndef OPTIMIZATION_LEVENBERG_MARQUARDT_H
#define OPTIMIZATION_LEVENBERG_MARQUARDT_H

#include "algebra/matrix_identity.h"
#include "levenberg_marquardt_logger.h"

template
<
typename Real,
typename DampingMatrix,
Verbosity V = Verbosity::SILENT,
std::ostream &stream = std::cout
>
class LevenbergMarquardt
{

    using Logger = LevenbergMarquardtLogger<V, stream>;

public:


    LevenbergMarquardt(const DampingMatrix &D_)
        : tol(1e-5), max_iter(10000), lambda(10.0), maximum(1000.0), decrease(2.0),
          increase(3.0), threshold(1.0), D(D_), current_cost(0.0), step_count(0)
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
            auto C  = lambda * D + B;
            dx = inv(C) * g;
            Vector xnew(x - dx);
            Real new_cost = J.cost_function(xnew);

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
        step_count++;
        current_cost = cost;
        Logger::step(*this);
        return 0;
    }

    unsigned int & maximum_iterations()
    { return max_iter; }

    void maximum_iterations(unsigned int n)
    { max_iter = n; }

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

    friend class LevenbergMarquardtLogger<V, stream>;
private:

    Real current_cost, tol, lambda, maximum, increase, decrease, threshold;
    unsigned int max_iter, step_count;

    // Positive definite matrix defining the trust region sphere r < ||Mx||.
    DampingMatrix D;

};

/* template */
/* < */
/* typename... Args */
/* > */
/* class Logger<Verbosity::VERBOSE, LevenbergMarquardt, Args... A> */
/* { */

/*     using Derived = LevenbergMarquardt<A>; */

/*     void separator( ostream& stream, */
/*                     Index length ) */
/*     { */
/*         for (Index i = 0; i < length; i++) */
/*             stream << "-"; */
/*         stream << endl; */
/*     } */

/*     step(ostream& stream) */
/*     { */
/*         LevenbergMarquardt& derived = static_cast<LevenbergMarquardt&>(*this); */
/*         stream << std::setw(10) << derived.step; */
/*         stream << std::setw(10) << derived.cost; */
/*         stream << std::setw(10) << derived.lambda; */
/*     } */
/* }; */

#endif // OPTIMIZATION_LEVENBERG_MARQUARDT_H

