#ifndef TEST_TEST_TYPES_H
#define TEST_TEST_TYPES_H

#include <boost/mpl/list.hpp>
#include <archetypes/matrix_archetype.h>

using Archetype = invlib::Matrix<MatrixArchetype<double>>;
using matrix_types = boost::mpl::list<Archetype>;

#endif // TEST_TEST_TYPES_H
