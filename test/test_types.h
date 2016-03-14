#ifndef TEST_TEST_TYPES_H
#define TEST_TEST_TYPES_H

#include <boost/mpl/list.hpp>
#include <interfaces/arts_wrapper.h>

using ArtsDense  = invlib::Matrix<ArtsMatrix<Matrix>, invlib::Vector<ArtsVector>>;
using matrix_types = boost::mpl::list<ArtsDense>;

#endif // TEST_TEST_TYPES_H
