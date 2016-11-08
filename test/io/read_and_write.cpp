#ifndef BOOST_TEST_MODULE
#define BOOST_TEST_MODULE "IO, Read and Write"
#endif

#include <boost/test/included/unit_test.hpp>
#include "invlib/io/writers.h"
#include "invlib/io/readers.h"

using namespace invlib;

void read_and_write_test()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_m(1, 100);
    std::uniform_int_distribution<> dis_n(1, 100);

    size_t m = 5; // dis_m(gen);
    size_t n = 5; // dis_n(gen);

    // Generate random sparse matrix.

    using SparseMatrix = SparseBase<double, Representation::Coordinates>;
    auto A = SparseMatrix::random(m, n);

    std::cout << A << std::endl;
    // Write and read matrix.

    invlib::write_matrix_arts<SparseMatrix>("test.xml", A, Format::Binary);
    auto B = invlib::read_matrix_arts<SparseMatrix>("test.xml");

    BOOST_TEST(A == B, "Error writing and reading Arts binary format.");

    invlib::write_matrix_arts<SparseMatrix>("test.xml", A, Format::ASCII);
    B = invlib::read_matrix_arts<SparseMatrix>("test.xml");
}

BOOST_AUTO_TEST_CASE(read_and_write)
{
    size_t ntests = 10;
    for (unsigned int i = 0; i < ntests; i++)
    {
        read_and_write_test();
    }
}
