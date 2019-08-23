#ifndef _TEST_TEST_TYPES_H_
#define _TEST_TEST_TYPES_H_

#include <invlib/algebra.h>
#include <invlib/utility/tuple.h>
#include <invlib/types/matrix_archetype.h>
#include <invlib/types/blas.h>
#include <invlib/types/mkl.h>

namespace invlib {

    using NumericTypes = std::tuple<float, double>;

    template <typename T>
    using Archetype = Matrix<MatrixArchetype<T>>;
    using Archetypes = typename tuple::Map<Archetype, NumericTypes>::Type;

    // BLAS dense types.
    using BlasTypes = typename tuple::Map<BlasMatrix, NumericTypes>::Type;

    using matrix_types = typename tuple::Concatenate<
        Archetypes
    #ifdef BLAS_AVAILABLE
        , BlasTypes
    #endif
        >::Type;


    //// Mkl Sparse types.
    //template <typename T>
    //using MklCoordinates = MklSparse<T, Representation::Coordinates>;
    //template <typename T>
    //using MklCompressedColumns = MklSparse<T, Representation::CompressedColumns>;
    //template <typename T>
    //using MklCompressedRows = MklSparse<T, Representation::CompressedRows>;
    //template <typename T>
    //using MklHybrid = MklSparse<T, Representation::Hybrid>;

    //using MklTypes = typename tuple::Concatenate<
    //    tuple::Map<MklCoordinates, NumericTypes>::Type,
    //    tuple::Map<MklCompressedColumns, NumericTypes>::Type,
    //    tuple::Map<MklCompressedRows, NumericTypes>::Type,
    //    tuple::Map<MklHybrid, NumericTypes>::Type
    //    >::Type;
}

#endif // _TEST_TEST_TYPES_H_
