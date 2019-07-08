/** \file mpi/traits.h
 *
 * \brief Contains type traits for a generic MPI implementation.
 *
 */

#ifndef MPI_TRAITS_H
#define MPI_TRAITS_H

namespace invlib
{

// -------------- //
//  Storage Types //
// -------------- //

/*! Passed as second template argument to the MpiMatrix or MpiVector classes,
 * the ConstRef template template triggers the MpiMatrix to only hold a
 * const reference to an already existing matrix or vector.
 */
template
<
typename T
>
struct ConstRef
{
    using type = const T &;
};

/*! Passed as second template argument to the MpiMatrix or MpiVector classes,
 * the LValue template template triggers the MpiMatrix to hold an MpiMatrix as
 * lvalue. This is required if a distributed matrix should be maniulated
 * or created from scratch and not from an already locally existing matrix or vector.
 */
template
<
typename T
>
struct LValue
{
    using type = T;
};

// ------------------- //
//   MPI Data Types    //
// ------------------- //

template <typename T>
struct MpiDataType;

template <>
struct MpiDataType<double>
{
public:
    static constexpr MPI_Datatype name = MPI_DOUBLE;
};

template <>
struct MpiDataType<float>
{
public:
    static constexpr MPI_Datatype name = MPI_FLOAT;
};

// ---------------- //
//      MPI Type    //
// ---------------- //

// Forward declarations.

template
<
typename Base
>
class Vector;

template
<
typename LocalType,
template <typename> class StorageTrait
>
class MpiVector;

template
<
typename Base
>
class Matrix;

template
<
typename LocalType,
template <typename> class StorageTrait
>
class MpiMatrix;

// MPIType struct.

template
<
typename T1,
template <typename> class StorageType
>
struct MpiTypeStruct;

template
<
typename T1,
template <typename> class StorageType
>
struct MpiTypeStruct<Vector<T1>, StorageType>
{
public:
    using type = Vector<MpiVector<T1, StorageType>>;
};

template
<
typename T1,
template <typename> class StorageType
>
struct MpiTypeStruct<Matrix<T1>, StorageType>
{
public:
    using type = Matrix<MpiMatrix<T1, StorageType>>;
};

// Type alias.

template
<
typename T1,
template <typename> class StorageType
>
using MpiType = typename MpiTypeStruct<T1, StorageType>::type;

}      // namespace invlib
#endif // MPI_TRATIS_H
