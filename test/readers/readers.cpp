
#include "invlib/interfaces/eigen.h"
#include "invlib/io/readers.h"

int main()
{
    std::string filename("vec.xml");
    invlib::read_matrix_arts<invlib::EigenSparse>("SeInv_bin.xml");
}
