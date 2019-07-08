#include <string.h>

// ---------------- //
//    MPI Vector    //
// ---------------- //

template
<
typename LocalType,
template <typename> class StorageType
>
MpiVector<LocalType, StorageType>::MpiVector()
    : local(), local_rows(0)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int *proc_rows = new int[nprocs];
    broadcast_local_rows(proc_rows);

    row_indices.reserve(nprocs);
    row_ranges.reserve(nprocs);

    for (unsigned int i = 0; i < nprocs; i++)
    {
        row_indices.push_back(0);
        row_ranges.push_back(0);
    }

    m = 0;
}

template
<
typename LocalType,
template <typename> class StorageType
>
   template<typename T, typename>
MpiVector<LocalType, StorageType>::MpiVector(T && local_vector)
    : local(local_vector)
{
    local_rows = local.rows();
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   int *proc_rows = new int[nprocs];
   broadcast_local_rows(proc_rows);

   unsigned int index = 0;
   row_indices.reserve(nprocs);
   row_ranges.reserve(nprocs);

   for (unsigned int i = 0; i < nprocs; i++)
   {
       row_indices.push_back(index);
       row_ranges.push_back(proc_rows[i]);
       index += proc_rows[i];
   }

   m = index;
}

template
<
typename LocalType,
template <typename> class StorageType
>
auto MpiVector<LocalType, StorageType>::split(const LocalType& v)
    -> MpiVector<LocalType, LValue>
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MpiVector<LocalType, LValue> w;
    w.resize(v.rows());

    w.get_local() = v.get_block(w.row_indices[rank], w.row_ranges[rank]);

    return w;
}

template
<
typename LocalType,
template <typename> class StorageType
>
auto MpiVector<LocalType, StorageType>::resize(unsigned int i)
    -> void
{
    m = i;

    // Distribute rows evenly over MPI processes.
    unsigned int total_rows = m;
    local_rows = total_rows / nprocs;
    unsigned int remainder = total_rows % nprocs;
    unsigned int local_start = local_rows * rank;

    if (rank < remainder) {
        local_rows += 1;
        local_start += rank;
    } else {
        local_start += remainder;
    }

    int *proc_rows = new int[nprocs];
    broadcast_local_rows(proc_rows);

    unsigned int index = 0;
    row_indices.reserve(nprocs);
    row_ranges.reserve(nprocs);

    for (unsigned int k = 0; k < nprocs; ++k) {
        row_indices[k] = index;
        row_ranges[k]  = proc_rows[k];
        index += proc_rows[k];
    }


    local.resize(local_rows);
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiVector<LocalType, StorageTemplate>::rows() const
    -> unsigned int
{
    return m;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiVector<LocalType, StorageTemplate>::get_local()
    -> LocalType &
{
    return local;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiVector<LocalType, StorageTemplate>::get_local() const
    -> const LocalType &
{
    return local;
}


template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiVector<LocalType, StorageTemplate>::operator()(unsigned int i) const
    -> RealType
{
    int owner;
    for (int r = 0; r < nprocs; r++)
    {
        if ((i >= row_indices[r]) && (i < row_indices[r] + row_ranges[r]))
            owner = r;
    }

    if (rank == owner)
        local_element = local(i - row_indices[rank]);

    MPI_Bcast(&local_element, 1, mpi_data_type, owner, MPI_COMM_WORLD);

    if (rank == owner)
        return local(i - row_indices[rank]);
    else
        return local_element;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiVector<LocalType, StorageTemplate>::operator()(unsigned int i)
    -> RealType &
{
    int owner = 0;
    for (int r = 0; r < nprocs; r++)
    {
        if ((i >= row_indices[r]) && (i < row_indices[r] + row_ranges[r]))
            owner = r;
    }

    if (rank == owner)
        local_element = local(i - row_indices[rank]);

    MPI_Bcast(&local_element, 1, mpi_data_type, owner, MPI_COMM_WORLD);

    if (rank == owner)
        return local(i - row_indices[rank]);
    else
        return local_element;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiVector<LocalType, StorageTemplate>::gather() const
    -> LocalType
{
    LocalType v; v.resize(m);
    broadcast_local_block(v.data_pointer(), local.data_pointer());
    MPI_Barrier(MPI_COMM_WORLD);
    return v;
}


template
<
typename LocalType,
template <typename> class StorageType
>
MpiVector<LocalType, StorageType>::operator LocalType() const
{
    LocalType v; v.resize(m);
    broadcast_local_block(v.data_pointer(), local.data_pointer());
    return v;
}

template
<
typename LocalType,
template <typename> class StorageType
>
auto MpiVector<LocalType, StorageType>::broadcast_local_rows(int rows[]) const
    -> void
{
    rows[rank] = local_rows;
    for (unsigned int i = 0; i < nprocs; i++)
    {
        MPI_Bcast(rows + i, 1, MPI_INTEGER, i, MPI_COMM_WORLD);
    }
}

template
<
typename LocalType,
template <typename> class StorageType
>
auto MpiVector<LocalType, StorageType>::broadcast_local_block(RealType *vector,
                                                              const RealType *block) const
    -> void
{
    memcpy(vector + row_indices[rank], block, row_ranges[rank] * sizeof(RealType));
    for (unsigned int i = 0; i < nprocs; i++)
    {
        MPI_Bcast(vector + row_indices[i], row_ranges[i], mpi_data_type,
                  i, MPI_COMM_WORLD);
    }
}

template
<
typename LocalType,
template <typename> class StorageType
>
auto MpiVector<LocalType, StorageType>::accumulate(const MpiVector &v)
    -> void
{
    local.accumulate(v.local);
}

template
<
typename LocalType,
template <typename> class StorageType
>
auto MpiVector<LocalType, StorageType>::subtract(const MpiVector &v)
    -> void
{
    local.subtract(v.local);
}

template
<
typename LocalType,
template <typename> class StorageType
>
auto MpiVector<LocalType, StorageType>::scale(RealType c)
    -> void
{
    local.scale(c);
}

template
<
typename LocalType,
template <typename> class StorageType
>
auto MpiVector<LocalType, StorageType>::norm() const
    -> RealType
{
    return sqrt(dot(*this, *this));
}

// ---------------- //
//    Dot Product   //
// ---------------- //

template
<
    typename T1,
    template <typename> class StorageType
>
auto dot(const MpiVector<T1, StorageType> &v, const MpiVector<T1, StorageType> &w)
    -> typename MpiVector<T1, StorageType>::RealType
{
    using RealType = typename MpiVector<T1, StorageType>::RealType;
    RealType local = dot(v.local, w.local);
    return mpi_sum(local);
}

template
<
    typename T1,
    template <typename> class StorageType
>
auto dot(const T1 &v, const MpiVector<T1, StorageType> &w)
    -> typename MpiVector<T1, StorageType>::RealType
{
    using RealType = typename MpiVector<T1, StorageType>::RealType;
    RealType local = dot(v.get_block(w.row_indices[w.rank], w.row_ranges[w.rank]),
                         w.local);
    return mpi_sum(local);
}

template
<
    typename T1,
    template <typename> class StorageType
>
auto dot(const MpiVector<T1, StorageType> &v, const T1 &w)
    -> typename MpiVector<T1, StorageType>::RealType
{
    using RealType = typename MpiVector<T1, StorageType>::RealType;
    RealType local = dot(w.get_block(v.row_indices[v.rank], v.row_ranges[v.rank]),
                         v.local);
    return mpi_sum(local);
}
