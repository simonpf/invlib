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
    std::uniform_int_distribution<> dis_m(1, 1000);
    std::uniform_int_distribution<> dis_n(1, 1000);

    size_t m = 4; dis_m(gen);
    size_t n = 4; dis_n(gen);

    // Generate random sparse matrix.

    using SparseMatrix = SparseData<double, int, Representation::Coordinates>;
    auto A = SparseMatrix::random(m, n);

    // Write and read matrix.

    invlib::write_matrix_arts("test_sparse_binary.xml", A, Format::Binary);
    auto B = invlib::read_matrix_arts("test_sparse_binary.xml");

    BOOST_TEST(A == B, "Error writing and reading Arts binary sparse matrix format.");

    invlib::write_matrix_arts("test_sparse_ascii.xml", A, Format::ASCII);
    B = invlib::read_matrix_arts("test_sparse_ascii.xml");


    BOOST_TEST(A == B, "Error writing and reading Arts ASCII sparse matrix format.");

    using Vector = VectorData<double>;
    auto v = Vector::random(n);

    // Write and read vector.

    invlib::write_vector_arts("test_vector_binary.xml", v, Format::Binary);
    auto w = invlib::read_vector_arts("test_vector_binary.xml");

    BOOST_TEST(v == w, "Error writing and reading Arts binary vector format.");

    invlib::write_vector_arts("test_vector_ascii.xml", v, Format::ASCII);
    w = invlib::read_vector_arts("test_vector_ascii.xml");

    BOOST_TEST(v == w, "Error writing and reading Arts ASCII vector format.");
}

BOOST_AUTO_TEST_CASE(read_and_write)
{
    size_t ntests = 100;
    for (unsigned int i = 0; i < ntests; i++)
    {
        read_and_write_test();
    }
}
