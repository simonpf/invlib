#define BOOST_TEST_MODULE algebra transformation
#include <boost/test/included/unit_test.hpp>
#include "algebra.h"
#include "algebra/Eigen.h"
#include "utility.h"
#include "test_types.h"

constexpr double EPS = 1e-10;
constexpr double zero = 0.0;
constexpr unsigned int ntests = 100;

template
<
typename T
>
void transformation_test(unsigned int n)
{

    using Real   = typename T::Real;
    using Vector = typename T::VectorBase;
    using Matrix = typename T::MatrixBase;

    auto A = random_positive_definite<Matrix>(n);
    auto B = random_positive_definite<Matrix>(n);
    auto v = random<Vector>(n);

    Real error;

    // Identity Transformation
    Identity I{};
    Vector w1 = I.apply(A * B) * I.apply(v);
    Vector w2 = A * B * v;
    error = maximum_error(w1, w2);
    BOOST_TEST((error < EPS),"maximum_error(w1, w2) = " << error);

    // NormalizeDiagonal Transform
    NormalizeDiagonal<Matrix> t(A);
    w1 = t.apply(inv(t.apply(A)) * t.apply(v));
    w2 = inv(A) * v;
    error = maximum_error(w1, w2);
    BOOST_TEST((error < EPS),"maximum_error(w1, w2) = " << error);

}

BOOST_AUTO_TEST_CASE_TEMPLATE(linear, T, matrix_types)
{
    srand(time(NULL));
    for (unsigned int i = 0; i < ntests; i++)
    {
        unsigned int n = 1 + rand() % 100;
        transformation_test<T>(n);
    }
}
