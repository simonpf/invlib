/** \file optimization/line_search.h
 *
 * \brief Contains functions for performing line search for a given
 * descend direction.
 *
 */

#ifndef OPTIMIZATION_LINE_SEARCH_H
#define OPTIMIZATION_LINE_SEARCH_H

namespace invlib
{

template
<
typename CostFunction,
typename VectorType,
typename RealType = typename VectorType::RealType
>
auto line_search(CostFunction &J,
                 const VectorType &x,
                 const VectorType &dx,
                 RealType &current_cost,
                 RealType start_value = 1.0,
                 RealType decrease    = 100.0)
    -> RealType
{
    RealType c = start_value;
    RealType new_cost = J.cost_function(x + c * dx);
    while (current_cost < new_cost)
    {
        c /= decrease;
        std::cout << "Line search: " << c << std::endl;
        new_cost = J.cost_function(x + c * dx);
    }
    current_cost = new_cost;
    return c;
}

}      // namespace invlib

#endif // OPTIMIZATION_LINE_SEARCH_H
