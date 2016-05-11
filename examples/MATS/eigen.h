#include <Eigen/Sparse>
#include <utility>

#include "invlib/traits.h"

using EigenSparseBase   = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using EigenVectorBase   = Eigen::VectorXd;

class EigenSparse;

// ----------------- //
//   Eigen Vector    //
// ----------------- //

class EigenVector : public EigenVectorBase
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

    template
    <
        typename T//,
        //typename = invlib::enable_if<invlib::is_constructible<EigenVectorBase, T>>
    >
    EigenVector(T &&t)
        : EigenVectorBase(std::forward<T>(t))
    {
        // Nothing to do here.
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

};

double dot(const EigenVector &v, const EigenVector &w)
{
    return v.dot(w);
}

// ----------------- //
//   Eigen Sparse    //
// ----------------- //

class EigenSparse : public EigenSparseBase
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

    template
    <
    typename T
    >
    EigenSparse(T &&t)
        : EigenSparseBase(std::forward<T>(t))
    {
        // Nothing to do here.
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
