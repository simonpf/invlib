#include <string.h>
#include <type_traits>

template
<
typename LocalType,
template <typename> class StorageTemplate
>
MpiMatrix<LocalType, StorageTemplate>::MpiMatrix()
    : local(), local_rows(0)
{
    static_assert(!is_same_template<StorageTemplate, ConstRef>::value,
                  "Default constructor not supported for reference "
                  "storage type.");

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    unsigned int index = 0;
    row_indices.reserve(nprocs);
    row_ranges.reserve(nprocs);

    for (unsigned int i = 0; i < nprocs; i++)
    {
        row_indices.push_back(0);
        row_ranges.push_back(0);
    }

    m = 0;
    n = 0;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::operator=(const MpiMatrix &A)
    -> MpiMatrix &
{
    local = A.local;
    local_rows = A.local_rows;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int *proc_rows = new int[nprocs];
    broadcast_local_rows(proc_rows);

    unsigned int index = 0;
    row_indices.clear();
    row_indices.reserve(nprocs);
    row_ranges.clear();
    row_ranges.reserve(nprocs);

    for (unsigned int i = 0; i < nprocs; i++)
    {
        row_indices.push_back(index);
        row_ranges.push_back(proc_rows[i]);
        index += proc_rows[i];
    }

    m = index;
    n = local.cols();
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
template<typename T, typename, typename>
MpiMatrix<LocalType, StorageTemplate>::MpiMatrix(T &&local_matrix)
    : local(std::forward<T>(local_matrix)),
      local_rows(remove_reference_wrapper(local_matrix).rows())
{
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
    n = remove_reference_wrapper(local).cols();
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
MpiMatrix<LocalType, StorageTemplate>::MpiMatrix(const LocalType &local_matrix)
    : local(local_matrix), local_rows(remove_reference_wrapper(local).rows())
{
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
    n = remove_reference_wrapper(local).cols();
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::split_matrix(const MatrixType &local_matrix)
    -> MpiMatrix<LocalType, LValue>
{
    int rank;
    int nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Distribute rows evenly over MPI processes.
    unsigned int total_rows = local_matrix.rows();
    unsigned int local_rows = total_rows / nprocs;
    unsigned int remainder = total_rows % nprocs;
    unsigned int local_start = local_rows * rank;


    if (rank < remainder)
    {
        local_rows += 1;
        local_start += rank;
    }
    else
    {
        local_start += remainder;
    }

    unsigned int n = local_matrix.cols();
    LocalType block = local_matrix.get_block(local_start, 0, local_rows, n);
    MpiMatrix<LocalType, LValue> splitted_matrix(block);
    return splitted_matrix;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::resize(unsigned int i,
                                                   unsigned int j)
      -> void
{
    m = i; n = j;

    // Distribute rows evenly over MPI processes.
    unsigned int total_rows = m;
    local_rows = total_rows / nprocs;
    unsigned int remainder = total_rows % nprocs;
    unsigned int local_start = local_rows * rank;

    if (rank < remainder)
    {
        local_rows += 1;
        local_start += rank;
    }
    else
    {
        local_start += remainder;
    }

    int *proc_rows = new int[nprocs];
    broadcast_local_rows(proc_rows);

    unsigned int index = 0;
    row_indices.reserve(nprocs);
    row_ranges.reserve(nprocs);

    for (unsigned int k = 0; k < nprocs; k++)
    {
        row_indices[k] = index;
        row_ranges[k]  = proc_rows[k];
        index += proc_rows[k];
    }

    local.resize(local_rows, j);
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::distribute(LocalType &local)
    -> void
{
    int m = local.rows();
    int n = local.cols();

    MPI_Bcast(&m, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

    MPI_Bcast(local.data_pointer(), m * n, mpi_data_type, 0, MPI_COMM_WORLD);
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
size_t MpiMatrix<LocalType, StorageTemplate>::rows() const
{
    return m;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
size_t MpiMatrix<LocalType, StorageTemplate>::cols() const
{
    return n;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::row(size_t i) const
    -> NonMpiVectorType
{
    NonMpiVectorType r(static_cast<MpiVectorType<LValue>>(local.row(i)));
    return r;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::col(size_t i) const
    -> MpiVectorType<LValue>
{
    MpiVectorType<LValue> c(local.col(i));
    return c;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::diagonal() const
    -> MpiVectorType<LValue>
{
    MpiVectorType<LValue> c(local.diagonal());
    return c;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::get_local()
    -> LocalType &
{
    return remove_reference_wrapper(local);
}


template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::operator()(unsigned int i,
                                                       unsigned int j) const
    -> RealType
{
    int owner;
    for (int r = 0; r < nprocs; r++)
    {
        if ((i >= row_indices[r]) && (i < row_indices[r] + row_ranges[r]))
            owner = r;
    }

    if (rank == owner)
        local_element = local(i - row_indices[rank], j);

    MPI_Bcast(&local_element, 1, mpi_data_type, owner, MPI_COMM_WORLD);

    if (rank == owner)
        return local(i - row_indices[rank], j);
    else
        return local_element;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::operator()(unsigned int i,
                                                       unsigned int j)
    -> RealType &
{
    int owner = 0;
    for (int r = 0; r < nprocs; r++)
    {
        if ((i >= row_indices[r]) && (i < row_indices[r] + row_ranges[r]))
            owner = r;
    }

    if (rank == owner)
        local_element = local(i - row_indices[rank], j);

    MPI_Bcast(&local_element, 1, mpi_data_type, owner, MPI_COMM_WORLD);

    if (rank == owner)
        return local(i - row_indices[rank], j);
    else
        return local_element;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
template <typename T>
auto MpiMatrix<LocalType, StorageTemplate>::multiply(T &v) const
    -> typename T::ResultType
{
    using ResultType = typename T::ResultType;
    ResultType w{}; w.resize(m);
    ResultType w_local{}; w_local.resize(local_rows);
    std::cout << "non-MPIVector multiply." << std::endl;
    w_local = remove_reference_wrapper(local).multiply(v);
    broadcast_local_block(w.get_element_pointer(), w_local.get_element_pointer());
    return w;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
template <template <typename> class VectorStorageTemplate>
auto MpiMatrix<LocalType, StorageTemplate>
::multiply(const MpiVectorType<VectorStorageTemplate> &v) const
    -> MpiVectorType<LValue>
{
    auto vv = v.gather();
    MpiVectorType<LValue> w = remove_reference_wrapper(local).multiply(vv);
    return w;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
template <typename T>
auto MpiMatrix<LocalType, StorageTemplate>::transpose_multiply(T &v) const
    -> typename T::ResultType
{
    using ResultType = typename T::ResultType;
    ResultType w_local{};   w_local.resize(n);
    ResultType w{};   w.resize(n);

    w_local = remove_reference_wrapper(local).transpose_multiply_block(v,
                                                                       row_indices[rank],
                                                                       row_ranges[rank]);
    reduce_vector_sum(w.get_element_pointer(), w_local.get_element_pointer());
    return w;
}

template
<
typename LocalType,
template <typename> class StorageTemplate
>
template <template <typename> class VectorStorageTemplate>
auto MpiMatrix<LocalType, StorageTemplate>
    ::transpose_multiply(const MpiVectorType<VectorStorageTemplate> &v) const
    -> MpiVectorType<LValue>
{
    NonMpiVectorType w; w.resize(n);
    NonMpiVectorType w_local;
    w_local = remove_reference_wrapper(local).transpose_multiply(v.get_local());
    reduce_vector_sum(w.get_element_pointer(), w_local.get_element_pointer());

    return MpiVectorType<LValue>::split(w);
}
// template
// <
// typename LocalType,
// template <typename> class StorageTemplate
// >
// MpiMatrix<LocalType, StorageTemplate>::operator MpiMatrix<LocalType, ConstRef>() const
// {
//     return MpiMatrix<LocalType, ConstRef>(local);
// }

// template
// <
// typename LocalType,
// template <typename> class StorageTemplate
// >
// MpiMatrix<LocalType, StorageTemplate>::operator LocalType()
// {
//     LocalType A; A.resize(m, n);
//     auto matrix_buffer = A.data_pointer();
//     auto start  = matrix_buffer + row_indices[rank] * n;
//     auto length = row_ranges[rank] * n;

//     std::copy(local.data_pointer(), local.data_pointer() + length, start);

//     for (int i = 0; i < nprocs; i++)
//     {
//         start  = matrix_buffer + row_indices[i] * n;
//         length = row_ranges[i] * n;
//         MPI_Bcast(start, length, mpi_data_type, i, MPI_COMM_WORLD);
//     }
//     return A;
// }

template
<
typename LocalType,
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::broadcast_local_rows(int *rows) const
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
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::broadcast_local_block(RealType * vector,
                                                                  const RealType * block) const
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
template <typename> class StorageTemplate
>
auto MpiMatrix<LocalType, StorageTemplate>::reduce_vector_sum(RealType *result_vector,
                                                              RealType *local_vector) const
    -> void
{
    memset(result_vector, 0, n * sizeof(RealType));
    MPI_Allreduce(local_vector, result_vector, n, mpi_data_type,
                  MPI_SUM, MPI_COMM_WORLD);
}
