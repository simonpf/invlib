#include "invlib/interfaces/eigen.h"
#include "invlib/io/writers.h"

#include <iostream>

using namespace invlib;

int main()
{
    EigenSparse matrix; matrix.resize(10, 10);

    std::vector<Eigen::Triplet<double>> triplets(10);

    for (size_t i = 0; i < 10; i++)
    {
        triplets.emplace_back(i, i, static_cast<double>(i));
    }
    triplets.emplace_back(4, 0, 3.142);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << matrix << std::endl;
    write_matrix_arts("test_bin.xml", matrix, Format::Binary);
    write_matrix_arts("test.xml",     matrix, Format::ASCII);
}
