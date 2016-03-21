#ifndef TEST_TEST_TYPES_H
#define TEST_TEST_TYPES_H

#include <boost/mpl/list.hpp>
#include <archetypes/matrix_archetype.h>
#include <interfaces/arts_wrapper.h>

using Archetype = invlib::Matrix<MatrixArchetype<double>>;
using Arts      = invlib::Matrix<ArtsMatrix>;
using matrix_types = boost::mpl::list<Arts>;

#endif // TEST_TEST_TYPES_H
