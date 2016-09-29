#ifndef BOOST_TEST_MODULE
#define BOOST_TEST_MODULE "Algebra, Diagonal"
#endif

#include <boost/test/included/unit_test.hpp>
#include "invlib/algebra.h"
#include "utility.h"
#include "test_types.h"

using namespace invlib;

template
<
typename MatrixType
>
void diagonal_test(unsigned int m)
{
    using VectorType = typename MatrixType::VectorType;

    MatrixType A = random_diagonal<MatrixType>(m);
    MatrixType B = random_diagonal<MatrixType>(m);
    MatrixType C = random_diagonal<MatrixType>(m);

    VectorType v, w;

    C = A * B;
    v = C.diagonal();
    w = (A * B).diagonal();
    double error = maximum_error(v, w);
    BOOST_TEST((error < EPS), "Product: " << m << ", error = " << error);

    C = A + B;
    v = C.diagonal();
    w = (A + B).diagonal();
    error = maximum_error(v, w);
    BOOST_TEST((error < EPS), "Sum: " << m << ", error = " << error);

    C = transp(A) * B * A + 3.0 * A;
    v = C.diagonal();
    w = (transp(A) * B * A + 3.0 * A).diagonal();
    error = maximum_error(v, w);
    BOOST_TEST((error < EPS), "OEM: " << m << ", error = " << error);

    C = (A * ((A * B) + (B * B) - A));
    v = C.diagonal();
    w = (A * ((A * B) + (B * B) - A)).diagonal();
    error = maximum_error(v, w);
    BOOST_TEST((error < EPS), "Complex Expression: " << m << ", error = " << error);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(diagonal,
                              T,
                              matrix_types)
{
    srand(time(NULL));
    for (unsigned int i = 0; i < ntests; i++)
    {
        unsigned int m = 1 + rand() % 100;
        diagonal_test<T>(m);
    }
}
