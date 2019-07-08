/** \file mpi/mpi_vector.h
 *
 * \brief Contains the MpiVector class, a generic class for vectors distributed
 * row-wise over nodes.
 *
 */

#ifndef MPI_MPI_VECTOR_H
#define MPI_MPI_VECTOR_H

#include "mpi.h"
#include "invlib/mpi/traits.h"
#include "invlib/mpi/utility.h"

namespace invlib
{

// ---------------- //
//    MPI Vector    //
// ---------------- //

template
<
typename LocalType,
template <typename> class StorageTrait = ConstRef
>
class MpiVector
{

public:

    /*! The basic scalar type. */
    using RealType   = typename LocalType::RealType;
    /*! The basic vector type  */
    using VectorType = MpiVector;
    /*! The local Matrix type.  */
    using MatrixType = typename LocalType::MatrixType;
    /*!
     * Result type of an algebraic expression with MPIMatrix as right hand
     * operator.
     */
    using ResultType = MpiVector;
    /*! The type used to store the local vector. */
    using StorageType = typename StorageTrait<LocalType>::type;

    MpiVector();


    template
    <
    typename T,
    typename = enable_if<is_constructible<StorageType, T>>
    >
    MpiVector(T &&local_vector);

    static MpiVector<LocalType, LValue> split(const LocalType &);

    // ------------------- //
    //     Manipulation    //
    // ------------------- //

    void resize(unsigned int i);

    unsigned int rows() const;

    LocalType & get_local();
    const LocalType & get_local() const;

    size_t get_index() const {return row_indices[rank];}
    size_t get_range() const {return row_ranges[rank];}

    RealType operator()(unsigned int i) const;
    RealType& operator()(unsigned int i);

    LocalType gather() const;

    operator LocalType() const;

    // ------------------------- //
    //    Arithmetic Opertations //
    // ------------------------- //

    void accumulate(const MpiVector& v);
    void subtract(const MpiVector& v);
    void scale(RealType c);
    RealType norm() const;

    const RealType * get_element_pointer() const {
        return local.get_element_pointer();
    }

    RealType * get_element_pointer() {
        return local.get_element_pointer();
    }
    template <typename T1, template <typename> class StorageType>
    friend auto dot(
        const MpiVector<T1, StorageType> &,
        const MpiVector<T1, StorageType> &)
    -> typename MpiVector<T1, StorageType>::RealType;

    template <typename T1, template <typename> class StorageType>
    friend auto dot(
        const T1 &,
        const MpiVector<T1, StorageType> &)
    -> typename MpiVector<T1, StorageType>::RealType;

    template <typename T1, template <typename> class StorageType>
    friend auto dot(
        const MpiVector<T1, StorageType> &,
        const T1 &)
    -> typename MpiVector<T1, StorageType>::RealType;

private:

    static constexpr MPI_Datatype mpi_data_type = MpiDataType<RealType>::name;

    void broadcast_local_rows(int proc_rows[]) const;
    void broadcast_local_block(RealType *vector,
                               const RealType *block) const;

    std::vector<unsigned int> row_indices;
    std::vector<unsigned int> row_ranges;

    int rank, nprocs;

    unsigned int m;
    unsigned int local_rows;

    StorageType local;
    RealType local_element;

};


#include "mpi_vector.cpp"

}      // namespace invlib

#endif // MPI_MPI_MATRIX_H
