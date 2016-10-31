#include "invlib/interfaces/eigen.h"
#include "invlib/io/readers.h"

int main()
{
    std::string filename("vec.xml");
    auto mat = invlib::read_matrix_arts<invlib::EigenSparse>("SeInv_bin.xml");
    auto vec = invlib::read_vector_arts<invlib::EigenVector>("xa.xml");
    std::cout << mat << std::endl;
    std::cout << vec << std::endl;
}
