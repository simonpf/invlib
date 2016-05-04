#include <Eigen/Sparse>
#include <utility>

using EigenSparseBase   = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using EigenVectorBase   = Eigen::VectorXd;

class EigenSparse;

// ----------------- //
//   Eigen Vector    //
// ----------------- //

class EigenVector : protected EigenVectorBase
{

public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using BaseType   = EigenVectorBase;
    using RealType   = double;
    using VectorType = EigenVector;
    using MatrixType = EigenSparse;
    using ResultType = EigenVector;

    // -------------- //
    //  Constructors  //
    // -------------- //

    EigenVector() = default;

    template <typename T>
    EigenVector(T &&t)
        : EigenVectorBase(std::forward<T>(t))
    {
        // Nothing to do here.
    }

    // ------------------- //
    //     Manipulation    //
    // ------------------- //

    void resize(unsigned int n)
    {
        this->EigenVectorBase::resize((int) n);
    }

    // ---------------------- //
    //  Arithmetic Operations //
    // ---------------------- //

    void accumulate(const EigenVector& v)
    {
        *this += v;
    }

    void subtract(const EigenVector& v)
    {
        *this -= v;
    }

    void scale(RealType c)
    {
        *this *= c;
    }

    RealType norm() const
    {
        return this->EigenVectorBase::norm();
    }

    friend EigenSparse;
    friend double dot(const EigenVector &v, const EigenVector &w);

};

double dot(const EigenVector &v, const EigenVector &w)
{
    return v.dot(w);
}

// ----------------- //
//   Eigen Sparse    //
// ----------------- //

class EigenSparse : protected EigenSparseBase
{

public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using BaseType   = EigenSparseBase;
    using RealType   = double;
    using VectorType = EigenVector;
    using MatrixType = EigenSparse;
    using ResultType = EigenSparse;

    // -------------- //
    //  Constructors  //
    // -------------- //

    template<typename T>
    EigenSparse(T &&t)
        : EigenSparseBase(std::forward<T>(t))
    {
        // Nothing to do here.
    }

    // ------------------- //
    //     Manipulation    //
    // ------------------- //

    unsigned int rows() const
    {
        std::cout << "rows" << std::endl;
        std::cout << this << std::endl;
        return this->EigenSparseBase::rows();
    }

    unsigned int cols() const
    {
        return this->EigenSparseBase::cols();
    }

    // ---------------------- //
    //  Arithmetic Operations //
    // ---------------------- //

    VectorType multiply(const VectorType &v) const
    {
        VectorType w = *this * v;
        return w;
    }

    VectorType transpose_multiply(const VectorType &v) const
    {
        VectorType w = this->transpose() * v;
        return w;
    }

};
