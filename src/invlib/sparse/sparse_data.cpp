// --------------------------- //
//   Coordinate Representation //
// --------------------------- //

template <typename Real, typename Index>
auto SparseData<Real, Index, Representation::Coordinates>::random(
    Index m,
    Index n)
    -> SparseData
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> real_dis(-10,10);
    std::uniform_int_distribution<> m_dis(0, m-1);
    std::uniform_int_distribution<> n_dis(0, n-1);
    std::uniform_int_distribution<> nelements_dis(std::min(m, n), std::max(m , n));

    std::vector<Index> rows;
    std::vector<Index> columns;
    std::vector<Real>   elements;

    Index nelements = nelements_dis(gen);
    rows.reserve(nelements);
    columns.reserve(nelements);
    elements.reserve(nelements);

    for (Index i = 0; i < nelements; i++)
    {
        rows.push_back(m_dis(gen));
        columns.push_back(n_dis(gen));
        elements.push_back(real_dis(gen));
    }

    SparseData matrix(m, n);
    matrix.set(rows, columns, elements);
    return matrix;

}

template <typename Real, typename Index>
SparseData<Real, Index, Representation::Coordinates>::SparseData(Index m_, Index n_)
    : m(m_), n(n_), nnz(0), column_indices(nullptr), row_indices(nullptr), elements(nullptr)
{
    // Nothing to do here.
}

template <typename Real, typename Index>
SparseData<Real, Index, Representation::Coordinates>::SparseData(
    const std::vector<Index> & rows_,
    const std::vector<Index> & columns_,
    const std::vector<Real>   & elements_)
{
    // Indirectly sort elements.
    std::vector<Index> indices(rows_.size());
    for (Index i = 0; i < indices.size(); i++)
    {
        indices[i] = i;
    }

    auto comp = [&](Index i, Index j)
        {
            return ((rows_[i] < rows_[j])  ||
                    ((rows_[i] == rows_[j]) &&
                     (columns_[i] < columns_[j])));
        };
    std::sort(indices.begin(), indices.end(), comp);

    // Set size of the matrix to maximum row and column-indices.
    m = (*row_indices)[indices.back()];
    n = (*column_indices)[indices.back()];

    Index nelements = elements.size();

    column_indices = std::shared_ptr<Index *>(new (Index *), ArrayDeleter<Index *>());
    row_indices    = std::shared_ptr<Index *>(new (Index *), ArrayDeleter<Index *>());
    elements       = std::shared_ptr<Real *>(new (Real *), ArrayDeleter<Real *>());

    *column_indices = new Index[nelements];
    *row_indices    = new Index[nelements];
    *elements       = new Real[nelements];

    for (Index i = 0; i < nelements; i++)
    {
        (*column_indices)[i]  = columns_[indices[i]];
        (*row_indices)[i]     = rows_[indices[i]];
        (*elements)[i]        = elements_[indices[i]];
    }
}

template <typename Real, typename Index>
SparseData<Real, Index, Representation::Coordinates>::SparseData(
    const MatrixArchetype<Real> & matrix)
    : m(matrix.rows()), n(matrix.cols())
{
    nnz = 0;
    for (Index i = 0; i < m; i++)
    {
        for (Index j = 0; j < n; j++)
        {
            if (std::abs(matrix(i,j)) > 0.0)
            {
                nnz++;
            }
        }
    }

    column_indices = std::shared_ptr<Index *>(new (Index *), ArrayDeleter<Index *>());
    row_indices    = std::shared_ptr<Index *>(new (Index *), ArrayDeleter<Index *>());
    elements       = std::shared_ptr<Real *>(new (Real *), ArrayDeleter<Real *>());

    *column_indices = new Index[nnz];
    *row_indices    = new Index[nnz];
    *elements       = new Real[nnz];

    Index index = 0;
    for (Index i = 0; i < m; i++)
    {
        for (Index j = 0; j < n; j++)
        {
            if (std::abs(matrix(i,j)) > 0.0)
            {
                (*column_indices)[index] = j;
                (*row_indices)[index]    = i;
                (*elements)[index]       = matrix(i,j);
                index++;
            }
        }
    }
}

template <typename Real, typename Index>
SparseData<Real, Index, Representation::Coordinates>::SparseData(
    const SparseData<Real, Index, Representation::CompressedColumns> & matrix)
    : m(matrix.rows()), n(matrix.cols()), nnz(matrix.non_zeros())
{

    std::vector<Index> row_index_vector(nnz);
    std::vector<Index> column_index_vector(nnz);
    std::vector<Real>   element_vector(nnz);

    Index column_index = 0;
    for (Index i = 0; i < nnz; i++)
    {
        while ((column_index < n) && (matrix.get_column_start_pointer()[column_index] == i))
        {
            column_index++;
        }
        row_index_vector[i]    = matrix.get_row_index_pointer()[i];
        column_index_vector[i] = column_index - 1;
        element_vector[i]      = matrix.get_element_pointer()[i];
    }
    set(row_index_vector, column_index_vector, element_vector);
}

template <typename Real, typename Index>
SparseData<Real, Index, Representation::Coordinates>::SparseData(
    const SparseData<Real, Index, Representation::CompressedRows> & matrix)
    : m(matrix.rows()), n(matrix.cols()), nnz(matrix.non_zeros())
{

    std::vector<Index> row_index_vector(nnz);
    std::vector<Index> column_index_vector(nnz);
    std::vector<Real>   element_vector(nnz);

    Index row_index = 0;
    for (Index i = 0; i < nnz; i++)
    {
        while ((row_index < m) && (matrix.get_row_start_pointer()[row_index] == i))
        {
            row_index++;
        }
        row_index_vector[i]    = row_index - 1;
        column_index_vector[i] = matrix.get_column_index_pointer()[i];
        element_vector[i]      = matrix.get_element_pointer()[i];
    }
    set(row_index_vector, column_index_vector, element_vector);
}

template
<
typename Real,
typename Index
>
void SparseData<Real, Index, Representation::Coordinates>::set(
    const std::vector<Index> & rows_,
    const std::vector<Index> & columns_,
    const std::vector<Real>   & elements_)
{

    nnz = elements_.size();

    // Indirectly sort elements.
    std::vector<Index> indices(nnz);
    for (Index i = 0; i < indices.size(); i++)
    {
        indices[i] = i;
    }

    auto comp = [&](Index i, Index j)
        {
            return ((rows_[i] < rows_[j])  ||
                    ((rows_[i] == rows_[j]) && (columns_[i] < columns_[j])));
        };
    std::sort(indices.begin(), indices.end(), comp);

    // Make sure all indices are valid.

    if (nnz > 0)
    {
        assert(m > rows_[indices[nnz - 1]]);
        assert(n > columns_[indices[nnz - 1]]);
    }

    Index nelements = elements_.size();

    column_indices = std::shared_ptr<Index *>(new (Index *), ArrayDeleter<Index *>());
    row_indices    = std::shared_ptr<Index *>(new (Index *), ArrayDeleter<Index *>());
    elements       = std::shared_ptr<Real *>(new (Real *), ArrayDeleter<Real *>());

    *column_indices = new Index[nelements];
    *row_indices    = new Index[nelements];
    *elements       = new Real[nelements];

    for (Index i = 0; i < nelements; i++)
    {
        (*column_indices)[i] = columns_[indices[i]];
        (*row_indices)[i]    = rows_[indices[i]];
        (*elements)[i]       = elements_[indices[i]];
    }
}

template <typename Real, typename Index>
SparseData<Real, Index, Representation::Coordinates>::
operator SparseData<Real, Index, Representation::CompressedColumns>() const
{
    std::shared_ptr<Index *> column_starts   {new (Index *),
                                               ArrayDeleter<Index *>()};
    std::shared_ptr<Index *> row_indices_new {new (Index *),
                                               ArrayDeleter<Index *>()};
    std::shared_ptr<Real *>   elements_new    {new (Real *),
                                               ArrayDeleter<Real *>()};
    *column_starts   = new Index[n + 1];
    *row_indices_new = new Index[nnz];
    *elements_new    = new Real[nnz];

    // Indirectly sort elements.
    std::vector<Index> indices(nnz);
    for (Index i = 0; i < nnz; i++)
    {
        indices[i] = i;
    }

    auto comp = [&](Index i, Index j)
    {
        return (((*column_indices)[i] < (*column_indices)[j])  ||
                (((*column_indices)[i] == (*column_indices)[j]) &&
                 ((*row_indices)[i] < (*row_indices)[j])));
    };
    std::sort(indices.begin(), indices.end(), comp);

    for(Index i = 0; i < nnz; i++)
    {
        (*row_indices_new)[i] = (*row_indices)[indices[i]];
        (*elements_new)[i]    = (*elements)[indices[i]];
    }

    Index j = 0;
    for (Index i = 0; i < n; i++)
    {
        (*column_starts)[i] = j;
        while ((j < nnz) && (i == (*column_indices)[indices[j]]))
        {
            j++;
        }
    }
    (*column_starts)[n] = nnz;

    SparseData<Real, Index, Representation::CompressedColumns> matrix{
        m, n, nnz, row_indices_new, column_starts, elements_new
            };

    return matrix;
}

template <typename Real, typename Index>
SparseData<Real, Index, Representation::Coordinates>::
operator SparseData<Real, Index, Representation::CompressedRows>() const
{
    std::shared_ptr<Index *> row_starts{new (Index *), ArrayDeleter<Index *>()};
    *row_starts   = new Index[m + 1];

    Index j = 0;
    for (Index i = 0; i < m; i++)
    {
        (*row_starts)[i] = j;
        while ((j < nnz) && i == (*row_indices)[j])
        {
            j++;
        }
    }
    (*row_starts)[m] = nnz;

    SparseData<Real, Index, Representation::CompressedRows> matrix{
        m, n, nnz, row_starts, column_indices, elements
            };

    return matrix;
}

template <typename Real, typename Index>
SparseData<Real, Index, Representation::Coordinates>::
operator MatrixArchetype<Real>() const
{
    MatrixArchetype<Real> matrix; matrix.resize(m, n);

    for (Index i = 0; i < m; i++)
    {
        for (Index j = 0; j < n; j++)
        {
            matrix(i, j) = 0.0;
        }
    }

    for (Index i = 0; i < nnz; i++)
    {
        matrix((*row_indices)[i], (*column_indices)[i]) += (*elements)[i];
    }

    return matrix;
}

template <typename Real, typename Index>
bool SparseData<Real, Index, Representation::Coordinates>:: operator == (
    const SparseData & B) const
{
    bool equal = (m == B.rows()) && (n == B.cols());
    for (Index i = 0; i < nnz; i++)
    {
        Index row_index    = (*row_indices)[i];
        Index column_index = (*column_indices)[i];

        Real element_sum       = (*elements)[i];
        Real element_sum_other = B.get_element_pointer()[i];
        while (((*row_indices)[i + 1]    == row_index) &&
               ((*column_indices)[i + 1] == column_index) &&
               (B.get_row_index_pointer()[i]    == row_index) &&
               (B.get_column_index_pointer()[i] == column_index) &&
               (i < nnz - 1))
        {
            i++;
            element_sum       += (*elements)[i];
            element_sum_other += B.get_element_pointer()[i];
        }
        equal = equal && numerical_equality(element_sum, element_sum_other);
    }
    return equal;
}

/*! Print sparse matrix to output stream. */
template <typename Real, typename Index>
std::ostream & operator << (
    std::ostream & s,
    const SparseData<Real, Index, Representation::Coordinates>& matrix)
{
    s << "Sparse Matrix Data, Coordinate Representation:" << std::endl;
    s << "Row Indices:    [";
    for (Index i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_row_index_pointer()[i] << " ";
    }
    s << matrix.get_row_index_pointer()[matrix.non_zeros()-1] << "]" << std::endl;

    s << "Column Indices: [";
    for (Index i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_column_index_pointer()[i] << " ";
    }
    s << matrix.get_column_index_pointer()[matrix.non_zeros()-1] << "]" << std::endl;

    s << "Elements:       [";
    for (Index i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_element_pointer()[i] << " ";
    }
    s << matrix.get_element_pointer()[matrix.non_zeros()-1] << "]" << std::endl;
    return s;
}

// ------------------ //
// Compressed Columns //
// ------------------ //

template <typename Real, typename Index>
SparseData<Real, Index, Representation::CompressedColumns>::SparseData(
    Index m_, Index n_, Index nnz_,
    const std::shared_ptr<Index *> & row_indices_,
    const std::shared_ptr<Index *> & column_starts_,
    const std::shared_ptr<Real *>   & elements_)
    : m(m_), n(n_), nnz(nnz_), column_starts(column_starts_),
      row_indices(row_indices_), elements(elements_)
{
    // Nothing to do here.
}

/*! Print sparse matrix to output stream. */
template <typename Real, typename Index>
std::ostream & operator << (
    std::ostream & s,
    const SparseData<Real, Index, Representation::CompressedColumns>& matrix)
{
    s << "Sparse Matrix Data, Compressed Row Representation:" << std::endl;
    s << "Row Indices:   [";
    for (Index i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_row_index_pointer()[i] << ", ";
    }
    s << matrix.get_row_index_pointer()[matrix.non_zeros() - 1] << "]" << std::endl;

    s << "Column Starts: [";
    for (Index i = 0; i < matrix.cols() - 1; i++)
    {
        s << matrix.get_column_start_pointer()[i] << ", ";
    }
    s << matrix.get_column_start_pointer()[matrix.cols() - 1] << "]" << std::endl;

    s << "Elements:      [";
    for (Index i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_element_pointer()[i] << ", ";
    }
    s << matrix.get_element_pointer()[matrix.non_zeros() - 1] << "]" << std::endl;
    return s;
}

// --------------- //
// Compressed Rows //
// --------------- //

template <typename Real, typename Index>
SparseData<Real, Index, Representation::CompressedRows>::SparseData(
    Index m_, Index n_, Index nnz_,
    const std::shared_ptr<Index *> & row_starts_,
    const std::shared_ptr<Index *> & column_indices_,
    const std::shared_ptr<Real *>   & elements_)
    : m(m_), n(n_), nnz(nnz_), column_indices(column_indices_),
      row_starts(row_starts_), elements(elements_)
{
    // Nothing to do here.
}

/*! Print sparse matrix to output stream. */
template <typename Real, typename Index>
std::ostream & operator << (
    std::ostream &s,
    const SparseData<Real, Index, Representation::CompressedRows>& matrix)
{
    s << "Sparse Matrix Data, Compressed Row Representation:" << std::endl;
    s << "Row Starts:     [";
    for (Index i = 0; i < matrix.rows() - 1; i++)
    {
        s << matrix.get_row_start_pointer()[i] << " ";
    }
    s << matrix.get_row_start_pointer()[matrix.rows() - 1] << "]" << std::endl;

    s << "Column Indices: [";
    for (Index i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_column_index_pointer()[i] << " ";
    }
    s << matrix.get_column_index_pointer()[matrix.non_zeros() - 1] << "]" << std::endl;

    s << "Elements:       [";
    for (Index i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_element_pointer()[i] << " ";
    }
    s << matrix.get_element_pointer()[matrix.non_zeros() - 1] << "]" << std::endl;
    return s;
}
