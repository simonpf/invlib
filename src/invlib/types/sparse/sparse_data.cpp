// --------------------------- //
//   Coordinate Representation //
// --------------------------- //

template <typename Real, typename size_t>
auto SparseData<Real, size_t, Representation::Coordinates>::random(
    size_t m,
    size_t n)
    -> SparseData
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> real_dis(-10,10);
    std::uniform_int_distribution<> m_dis(0, m-1);
    std::uniform_int_distribution<> n_dis(0, n-1);
    std::uniform_int_distribution<> nelements_dis(std::min(m, n), std::max(m , n));

    std::vector<size_t> rows;
    std::vector<size_t> columns;
    std::vector<Real>   elements;

    size_t nelements = nelements_dis(gen);
    rows.reserve(nelements);
    columns.reserve(nelements);
    elements.reserve(nelements);

    for (size_t i = 0; i < nelements; i++)
    {
        rows.push_back(m_dis(gen));
        columns.push_back(n_dis(gen));
        elements.push_back(real_dis(gen));
    }

    SparseData matrix(m, n);
    matrix.set(rows, columns, elements);
    return matrix;

}

template <typename Real, typename size_t>
SparseData<Real, size_t, Representation::Coordinates>::SparseData(size_t m_, size_t n_)
    : m(m_), n(n_), nnz(0), column_indices(nullptr), row_indices(nullptr), elements(nullptr)
{
    // Nothing to do here.
}

template <typename Real, typename size_t>
SparseData<Real, size_t, Representation::Coordinates>::SparseData(
    const std::vector<size_t> & rows_,
    const std::vector<size_t> & columns_,
    const std::vector<Real>   & elements_)
{
    // Indirectly sort elements.
    std::vector<size_t> indices(rows_.size());
    for (size_t i = 0; i < indices.size(); i++)
    {
        indices[i] = i;
    }

    auto comp = [&](size_t i, size_t j)
        {
            return ((rows_[i] < rows_[j])  ||
                    ((rows_[i] == rows_[j]) &&
                     (columns_[i] < columns_[j])));
        };
    std::sort(indices.begin(), indices.end(), comp);

    // Set size of the matrix to maximum row and column-indices.
    m = (*row_indices)[indices.back()];
    n = (*column_indices)[indices.back()];

    size_t nelements = elements.size();

    column_indices = array::create(nelements);
    row_indices    = array::create(nelements);
    elements       = array::create(nelements);

    for (size_t i = 0; i < nelements; i++) {
        column_indices[i]  = columns_[indices[i]];
        row_indices[i]     = rows_[indices[i]];
        elements[i]        = elements_[indices[i]];
    }
}

template <typename Real, typename size_t>
SparseData<Real, size_t, Representation::Coordinates>::SparseData(
    const MatrixArchetype<Real> & matrix)
    : m(matrix.rows()), n(matrix.cols())
{
    nnz = 0;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            if (std::abs(matrix(i,j)) > 0.0) {
                nnz++;
            }
        }
    }

    column_indices = array::create<size_t>(nnz);
    row_indices    = array::create<size_t>(nnz);
    elements       = array::create<Real>(nnz);

    size_t index = 0;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            if (std::abs(matrix(i,j)) > 0.0) {
                column_indices[index] = j;
                row_indices[index]    = i;
                elements[index]       = matrix(i,j);
                index++;
            }
        }
    }
}

template <typename Real, typename size_t>
SparseData<Real, size_t, Representation::Coordinates>::SparseData(
    const SparseData<Real, size_t, Representation::CompressedColumns> & matrix)
    : m(matrix.rows()), n(matrix.cols()), nnz(matrix.non_zeros())
{

    std::vector<size_t> row_index_vector(nnz);
    std::vector<size_t> column_index_vector(nnz);
    std::vector<Real>   element_vector(nnz);

    size_t column_index = 0;
    for (size_t i = 0; i < nnz; i++)
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

template <typename Real, typename size_t>
SparseData<Real, size_t, Representation::Coordinates>::SparseData(
    const SparseData<Real, size_t, Representation::CompressedRows> & matrix)
    : m(matrix.rows()), n(matrix.cols()), nnz(matrix.non_zeros())
{

    std::vector<size_t> row_index_vector(nnz);
    std::vector<size_t> column_index_vector(nnz);
    std::vector<Real>   element_vector(nnz);

    size_t row_index = 0;
    for (size_t i = 0; i < nnz; i++)
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
typename size_t
>
void SparseData<Real, size_t, Representation::Coordinates>::set(
    const std::vector<size_t> & rows_,
    const std::vector<size_t> & columns_,
    const std::vector<Real>   & elements_)
{

    nnz = elements_.size();

    // Indirectly sort elements.
    std::vector<size_t> indices(nnz);
    for (size_t i = 0; i < static_cast<size_t>(indices.size()); i++) {
        indices[i] = i;
    }

    auto comp = [&](size_t i, size_t j) {
            return ((rows_[i] < rows_[j])  ||
                    ((rows_[i] == rows_[j]) && (columns_[i] < columns_[j])));
        };
    std::sort(indices.begin(), indices.end(), comp);

    // Make sure all indices are valid.

    if (nnz > 0) {
        assert(m > rows_[indices[nnz - 1]]);
        assert(n > columns_[indices[nnz - 1]]);
    }

    size_t nelements = elements_.size();

    column_indices = array::create<size_t>(nelements);
    row_indices    = array::create<size_t>(nelements);
    elements       = array::create<Real>(nelements);


    for (size_t i = 0; i < nelements; i++)
    {
        column_indices[i] = columns_[indices[i]];
        row_indices[i]    = rows_[indices[i]];
        elements[i]       = elements_[indices[i]];
    }
}

template <typename Real, typename size_t>
SparseData<Real, size_t, Representation::Coordinates>::
operator SparseData<Real, size_t, Representation::CompressedColumns>() const
{
    auto column_starts   = array::create<size_t>(n + 1);
    auto row_indices_new = array::create<size_t>(nnz);
    auto elements_new    = array::create<size_t>(nnz);

    // Indirectly sort elements.
    std::vector<size_t> indices(nnz);
    for (size_t i = 0; i < nnz; i++)
    {
        indices[i] = i;
    }

    auto comp = [&](size_t i, size_t j)
    {
        return ((column_indices[i] < column_indices[j])  ||
                ((column_indices[i] == column_indices[j]) &&
                 (row_indices[i] < row_indices[j])));
    };
    std::sort(indices.begin(), indices.end(), comp);

    for(size_t i = 0; i < nnz; i++)
    {
        (*row_indices_new)[i] = (*row_indices)[indices[i]];
        (*elements_new)[i]    = (*elements)[indices[i]];
    }

    size_t j = 0;
    for (size_t i = 0; i < n; i++)
    {
        column_starts[i] = j;
        while ((j < nnz) && (i == column_indices[indices[j]]))
        {
            j++;
        }
    }
    column_starts[n] = nnz;

    SparseData<Real, size_t, Representation::CompressedColumns> matrix{
        m, n, nnz, row_indices_new, column_starts, elements_new
            };

    return matrix;
}

template <typename Real, typename size_t>
SparseData<Real, size_t, Representation::Coordinates>::
operator SparseData<Real, size_t, Representation::CompressedRows>() const
{
    auto row_starts = array::create<size_t>(m + 1);

    size_t j = 0;
    for (size_t i = 0; i < m; i++)
    {
        (*row_starts)[i] = j;
        while ((j < nnz) && i == (*row_indices)[j])
        {
            j++;
        }
    }
    (*row_starts)[m] = nnz;

    SparseData<Real, size_t, Representation::CompressedRows> matrix{
        m, n, nnz, row_starts, column_indices, elements
            };

    return matrix;
}

template <typename Real, typename size_t>
SparseData<Real, size_t, Representation::Coordinates>::
operator MatrixArchetype<Real>() const
{
    MatrixArchetype<Real> matrix; matrix.resize(m, n);

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            matrix(i, j) = 0.0;
        }
    }

    for (size_t i = 0; i < nnz; i++)
    {
        matrix((*row_indices)[i], (*column_indices)[i]) += (*elements)[i];
    }

    return matrix;
}

template <typename Real, typename size_t>
bool SparseData<Real, size_t, Representation::Coordinates>:: operator == (
    const SparseData & B) const
{
    bool equal = (m == B.rows()) && (n == B.cols());
    for (size_t i = 0; i < nnz; i++)
    {
        size_t row_index    = row_indices[i];
        size_t column_index = column_indices[i];

        Real element_sum       = elements[i];
        Real element_sum_other = B.get_element_pointer()[i];
        while ((row_indices[i + 1]    == row_index) &&
               (column_indices[i + 1] == column_index) &&
               (B.get_row_index_pointer()[i]    == row_index) &&
               (B.get_column_index_pointer()[i] == column_index) &&
               (i < nnz - 1))
        {
            i++;
            element_sum       += elements[i];
            element_sum_other += B.get_element_pointer()[i];
        }
        equal = equal && numerical_equality(element_sum, element_sum_other);
    }
    return equal;
}

/*! Print sparse matrix to output stream. */
template <typename Real, typename size_t>
std::ostream & operator << (
    std::ostream & s,
    const SparseData<Real, size_t, Representation::Coordinates>& matrix)
{
    s << "Sparse Matrix Data, Coordinate Representation:" << std::endl;
    s << "Row Indices:    [";
    for (size_t i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_row_index_pointer()[i] << " ";
    }
    s << matrix.get_row_index_pointer()[matrix.non_zeros()-1] << "]" << std::endl;

    s << "Column Indices: [";
    for (size_t i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_column_index_pointer()[i] << " ";
    }
    s << matrix.get_column_index_pointer()[matrix.non_zeros()-1] << "]" << std::endl;

    s << "Elements:       [";
    for (size_t i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_element_pointer()[i] << " ";
    }
    s << matrix.get_element_pointer()[matrix.non_zeros()-1] << "]" << std::endl;
    return s;
}

// ------------------ //
// Compressed Columns //
// ------------------ //

template <typename Real, typename size_t>
SparseData<Real, size_t, Representation::CompressedColumns>::SparseData(
    size_t m_, size_t n_, size_t nnz_,
    const std::shared_ptr<size_t[]> & row_indices_,
    const std::shared_ptr<size_t[]> & column_starts_,
    const std::shared_ptr<Real[]>   & elements_)
    : m(m_), n(n_), nnz(nnz_), column_starts(column_starts_),
      row_indices(row_indices_), elements(elements_)
{
    // Nothing to do here.
}

/*! Print sparse matrix to output stream. */
template <typename Real, typename size_t>
std::ostream & operator << (
    std::ostream & s,
    const SparseData<Real, size_t, Representation::CompressedColumns>& matrix)
{
    s << "Sparse Matrix Data, Compressed Column Representation:" << std::endl;
    s << "Row Indices:   [";
    for (size_t i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_row_index_pointer()[i] << ", ";
    }
    s << matrix.get_row_index_pointer()[matrix.non_zeros() - 1] << "]" << std::endl;

    s << "Column Starts: [";
    for (size_t i = 0; i < matrix.cols() - 1; i++)
    {
        s << matrix.get_column_start_pointer()[i] << ", ";
    }
    s << matrix.get_column_start_pointer()[matrix.cols() - 1] << "]" << std::endl;

    s << "Elements:      [";
    for (size_t i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_element_pointer()[i] << ", ";
    }
    s << matrix.get_element_pointer()[matrix.non_zeros() - 1] << "]" << std::endl;
    return s;
}

// --------------- //
// Compressed Rows //
// --------------- //

template <typename Real, typename size_t>
SparseData<Real, size_t, Representation::CompressedRows>::SparseData(
    size_t m_, size_t n_, size_t nnz_,
    const std::shared_ptr<size_t[]> & row_starts_,
    const std::shared_ptr<size_t[]> & column_indices_,
    const std::shared_ptr<Real[]>   & elements_)
    : m(m_), n(n_), nnz(nnz_), column_indices(column_indices_),
      row_starts(row_starts_), elements(elements_)
{
    // Nothing to do here.
}

/*! Print sparse matrix to output stream. */
template <typename Real, typename size_t>
std::ostream & operator << (
    std::ostream &s,
    const SparseData<Real, size_t, Representation::CompressedRows>& matrix)
{
    s << "Sparse Matrix Data, Compressed Row Representation:" << std::endl;
    s << "Row Starts:     [";
    for (size_t i = 0; i < matrix.rows() - 1; i++)
    {
        s << matrix.get_row_start_pointer()[i] << " ";
    }
    s << matrix.get_row_start_pointer()[matrix.rows() - 1] << "]" << std::endl;

    s << "Column Indices: [";
    for (size_t i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_column_index_pointer()[i] << " ";
    }
    s << matrix.get_column_index_pointer()[matrix.non_zeros() - 1] << "]" << std::endl;

    s << "Elements:       [";
    for (size_t i = 0; i < matrix.non_zeros() - 1; i++)
    {
        s << matrix.get_element_pointer()[i] << " ";
    }
    s << matrix.get_element_pointer()[matrix.non_zeros() - 1] << "]" << std::endl;
    return s;
}
