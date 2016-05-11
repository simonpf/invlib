#include <Eigen/Sparse>

#include <string>
#include <fstream>

/*! Type alias for the Eigen sparse matrix type used
 * in the example.
 */
using EigenSparseBase = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using EigenVectorBase = Eigen::VectorXd;
using EigenIndex   = int;


/*! Read Eigen sparse matrix from file.
 *
 * The binary format used simply contains m, n and the number of
 * non-zero elements (nnz) as ints, followed by the nnz row indices,
 * the nnz column indices and the nnz elements given as doubles.
 *
 * \param filename File to read the matrix from.
 * \return The matrix read from the given file.
 */
EigenSparseBase read_sparse_matrix(std::string filename)
{
    std::ifstream file;
    file.open(filename, std::ios::binary);

    EigenIndex m, n, nnz;
    std::vector<EigenIndex> cols, rows;
    std::vector<double> elements;

    file.read((char*) &m, sizeof(EigenIndex));
    file.read((char*) &n, sizeof(EigenIndex));
    file.read((char*) &nnz, sizeof(EigenIndex));

    std::cout << "Reading matrix: " << m << " / " << n << " / " << nnz << std::endl;

    cols.resize(nnz);
    rows.resize(nnz);
    elements.resize(nnz);

    // Read row indices, columns indices and elements from file.

    file.read((char*) &rows[0], nnz * sizeof(EigenIndex));
    file.read((char*) &cols[0], nnz * sizeof(EigenIndex));
    file.read((char*) &elements[0], nnz * sizeof(double));

    // Convert vectors to triplet list.
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);

    for (int i = 0; i < nnz; i++)
    {
        triplets.push_back(Eigen::Triplet<double>(rows[i], cols[i], elements[i]));
    }

    file.close();

    EigenSparseBase mat(m, n); mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
}

/*! Store sparse matrix to file.
 *
 * Stores matrix in binary form. The format is
 *
 *     - First three integers of type int: m, n, nnz (number of non-zero elements)
 *     - Row indices as int
 *     - Column indices s
 *     - Elements as double
 *
 * \param mat The matrix to write to disk.
 * \param filename The name of file.
 */
void write_sparse_matrix(const EigenSparseBase &mat, std::string filename)
{
    std::fstream file;
    file.open(filename, std::ios::out | std::ios::binary | std::ios::trunc);

    EigenIndex m, n, nnz;

    m   = mat.rows();
    n   = mat.cols();
    nnz = mat.nonZeros();

    std::cout << "Writing " << m << " / " << n << " / " << nnz << std::endl;

    file.write((char*) &m, sizeof(EigenIndex));
    file.write((char*) &n, sizeof(EigenIndex));
    file.write((char*) &nnz, sizeof(EigenIndex));

    // Write row indices. Yes, it's that painful.

    const EigenIndex *indx = mat.outerIndexPtr();
    EigenIndex current = indx[0];
    int count = 0;
    EigenIndex tmp;

    for (EigenIndex i = 1; i < mat.outerSize(); i++)
    {
        tmp = i - 1;
        for (EigenIndex j = 0; j < indx[i] - current; j++)
        {
            count++;
            file.write((char*) &tmp, sizeof(EigenIndex));
        }
        current = indx[i];
    }

    m -= 1;
    for (EigenIndex j = count; j < nnz; j++)
    {
        count++;
        file.write((char *) &m, sizeof(EigenIndex));
    }

    // Write column indices.
    file.write((char*) mat.innerIndexPtr(), nnz * sizeof(EigenIndex));

    // Write elements.
    file.write((char*) mat.valuePtr(), nnz * sizeof(double));

    file.close();
}

/*! Read Eigen vector from file.
 *
 * \param filename File to read the vector from.
 * \return The vector read from the given file.
 */
EigenVectorBase read_vector(std::string filename)
{
    std::ifstream file;
    file.open(filename, std::ios::binary);

    EigenIndex n;
    file.read((char*) &n, sizeof(EigenIndex));

    std::cout << "reading vector: " << n << std::endl;
    EigenVectorBase v(n);
    file.read((char*) v.data(), n * sizeof(double));

    return v;
}

/*! Store vector to file.
 *
 * \param v The vector to write to disk.
 * \param filename The name of file.
 */
void write_vector(const EigenVectorBase &v, std::string filename)
{
    std::fstream file;
    file.open(filename, std::ios::out | std::ios::binary | std::ios::trunc);

    EigenIndex n = v.rows();
    file.write((char*) &n, sizeof(EigenIndex));
    file.write((char*) v.data(), n * sizeof(double));

    file.close();
}
