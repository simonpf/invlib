template
<
typename LocalType,
template <typename> typename StorageTemplate
>
template<typename T, typename>
MPIMatrix<LocalType, StorageTemplate>::MPIMatrix(T &&local_matrix)
    : local(std::forward<T>(local_matrix)), local_rows(local.rows())
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
    n = local.cols();
}

template
<
typename LocalType,
template <typename> typename StorageTemplate
>
auto MPIMatrix<LocalType, StorageTemplate>::split_matrix(const MatrixType &local_matrix)
    -> MPIMatrix<LocalType, LValue>
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
    MPIMatrix<LocalType, LValue> splitted_matrix =
        local_matrix.get_block(local_start, 0, local_rows, n);
    return splitted_matrix;
}

template
<
typename LocalType,
template <typename> typename StorageTemplate
>
auto MPIMatrix<LocalType, StorageTemplate>::broadcast(LocalType &local)
    -> void
{
    int m = local.rows();
    int n = local.cols();

    MPI_Bcast(&m, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

    MPI_Bcast(local.raw_pointer(), m * n, mpi_data_type, 0, MPI_COMM_WORLD);
}

template
<
typename LocalType,
template <typename> typename StorageTemplate
>
auto MPIMatrix<LocalType, StorageTemplate>::rows() const
    -> unsigned int
{
    return m;
}

template
<
typename LocalType,
template <typename> typename StorageTemplate
>
auto MPIMatrix<LocalType, StorageTemplate>::cols() const
    -> unsigned int
{
    return n;
}

template
<
typename LocalType,
template <typename> typename StorageTemplate
>
auto MPIMatrix<LocalType, StorageTemplate>::multiply(const VectorType &v) const
    -> VectorType
{
    VectorType w{}; w.resize(m);
    VectorType w_local{}; w_local.resize(local_rows);
    w_local = local.multiply(v);
    broadcast_local_blocks(w.raw_pointer(), w_local.raw_pointer());
    return w;
}

template
<
typename LocalType,
template <typename> typename StorageTemplate
>

auto MPIMatrix<LocalType, StorageTemplate>::transpose_multiply(const VectorType &v) const
    -> VectorType
{
    VectorType w_local{};   w_local.resize(n);
    VectorType w{};   w.resize(n);

    w_local = local.transpose_multiply_block(v, row_indices[rank], row_ranges[rank]);
    reduce_vector_sum(w.raw_pointer(), w_local.raw_pointer());
    return w;
}

template
<
typename LocalType,
template <typename> typename StorageTemplate
>
auto MPIMatrix<LocalType, StorageTemplate>::broadcast_local_rows(int rows[]) const
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
template <typename> typename StorageTemplate
>
auto MPIMatrix<LocalType, StorageTemplate>::broadcast_local_blocks(double *vector,
                                                  const double *block) const
    -> void
{
    memcpy(vector + row_indices[rank], block, row_ranges[rank] * sizeof(double));
    for (unsigned int i = 0; i < nprocs; i++)
    {
        MPI_Bcast(vector + row_indices[i], row_ranges[i], mpi_data_type,
                  i, MPI_COMM_WORLD);
    }
}

template
<
typename LocalType,
template <typename> typename StorageTemplate
>
auto MPIMatrix<LocalType, StorageTemplate>::reduce_vector_sum(double *result_vector,
                                             double *local_vector) const
    -> void
{
    memset(result_vector, 0, n * sizeof(double));
    MPI_Allreduce(local_vector, result_vector, n, mpi_data_type,
                  MPI_SUM, MPI_COMM_WORLD);
}
