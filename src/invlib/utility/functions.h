/** \file utility/reference_wrapper.h
 *
 * \brief Contains short, helpful functions that don't fit
 * in elsewhere.
 *
 */

#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

namespace invlib
{

/*! Numerical equality.
 * Check if two floating point values are equal up to machine precision.
 * The criterion used is the relative difference with respect to the largest
 * absolute value of the two floating point numbers, if this absolute value
 * is larger than zero (up to machine precision). Otherwise the absolute difference
 * is used.
 * The precision threshold are set to 1e-4 and 1e-9 for single and double precision
 * floating point numbers.
 *
 * \tparam The type of the floation point values to compare.
 */
template <typename T> bool numerical_equality(const T &, const T &);

// Single precision specialization.
template<>
bool numerical_equality<float>(const float & a, const float & b)
{
    float ref = std::max(std::abs(a), std::abs(b));

    if (ref > 1e-4)
    {
        return ((std::abs(a - b) / ref) < 1e-4);
    } else {
        return std::abs(a - b) < 1e-4;
    }
}

// Double precision specialization.
template<>
bool numerical_equality<double>(const double & a, const double & b)
{
    double ref = std::max(std::abs(a), std::abs(b));

    if (ref > 1e-4)
    {
        return ((std::abs(a - b) / ref) < 1e-9);
    } else {
        return std::abs(a - b) < 1e-9;
    }
}

}      // namespace invlib
#endif // UTILITY_FUNCTIONS_H
