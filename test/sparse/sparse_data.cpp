#ifndef BOOST_TEST_MODULE
#define BOOST_TEST_MODULE "Algebra, Identities"
#endif

#include <boost/test/included/unit_test.hpp>
#include "invlib/sparse/sparse_data.h"

using namespace invlib;

void conversions_test()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_m(1, 100);
    std::uniform_int_distribution<> dis_n(1, 100);

    size_t m = dis_m(gen);
    size_t n = dis_n(gen);

    auto A = SparseData<double, Representation::Coordinates>::random(m, n);
    MatrixArchetype<double> B(A);
    SparseData<double, Representation::Coordinates> C(B);
    SparseData<double, Representation::CompressedColumns> D1(C);
    SparseData<double, Representation::Coordinates> E1(D1);
    MatrixArchetype<double> F1(E1);
    MatrixArchetype<double> G = F1;
    G.subtract(B);

    double maximum_error = 0.0;
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            maximum_error = std::max(maximum_error, std::abs(G(i,j)));
        }
    }

    BOOST_TEST((maximum_error < 1e-4), "Error: Maximum difference = " << maximum_error);

    SparseData<double, Representation::CompressedRows> D2(C);
    SparseData<double, Representation::Coordinates> E2(B);
    MatrixArchetype<double> F2(E2);
    G = F2;
    G.subtract(B);

    maximum_error = 0.0;
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0;j < n; j++)
        {
            maximum_error = std::max(maximum_error, std::abs(G(i,j)));
        }
    }

    BOOST_TEST((maximum_error <  1e-4), "Error: Maximum difference = " << maximum_error);
}

BOOST_AUTO_TEST_CASE(conversions)
{
    srand(time(NULL));
    for (unsigned int i = 0; i < 100 /*ntests*/; i++)
    {
        conversions_test();
    }
}
