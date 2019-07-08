#ifndef _TEST_TEST_TYPES_H_
#define _TEST_TEST_TYPES_H_

#include <invlib/algebra.h>
#include <invlib/utility/tuple.h>
#include <invlib/types/matrix_archetype.h>

namespace invlib {

    template <typename T>
    using Archetype = Matrix<MatrixArchetype<T>>;
    using Archetypes = typename tuple::Map<Archetype, std::tuple<float, double>>::Type;

    using matrix_types = Archetypes;
}

#endif // _TEST_TEST_TYPES_H_
