#ifndef INTERFACES_ARTS_WRAPPER_H
#define INTERFACES_ARTS_WRAPPER_H

#define HAVE_CONFIG_H (1)

#include "matpack.h"
#include "matpackI.h"
#include "matpackII.h"
#include "lin_alg.h"

#include "invlib/traits.h"

using invlib::disable_if;
using invlib::is_same;
using invlib::decay;

class ArtsMatrix;

class ArtsVector : public Vector
{
public:

    using RealType = Numeric;
    using VectorType = ArtsVector;
    using MatrixType = ArtsMatrix;
    using ResultType = ArtsVector;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    ArtsVector() : Vector() {}
    ArtsVector(const ArtsVector &A) = default;
    ArtsVector & operator=(const ArtsVector &A) = default;

    ArtsVector(ArtsVector &&A)
    {
        this->mrange = A.mrange;
        this->mdata  = A.mdata;
        A.mdata      = nullptr;
    }

    ArtsVector & operator=(ArtsVector &&A)
    {
        delete[] this->mdata;
        this->mrange  = A.mrange;
        this->mdata   = A.mdata;
        A.mdata       = nullptr;
    }

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    Index rows() const {return this->nelem();}

    Numeric operator()(Index i) const
    {
        return this->get(i);
    }

    Numeric& operator()(Index i)
    {
        return this->get(i);
    }

    void accumulate(const ArtsVector& w)
    {
        this->operator+=(w);
    }

    void subtract(const ArtsVector& w)
    {
        this->operator-=(w);
    }

    void scale(Numeric c)
    {
        this->operator*=(c);
    }

    Numeric norm()
    {
        return sqrt(operator*(*this, *this));
    }

};

/** \brief Arts dense matrix interace wrapper.
 *
 * Simple wrapper class providing an interface to the ARTS matrix class.
 *
 */
class ArtsMatrix : public Matrix
{
public:

    using RealType = Numeric;
    using VectorType = ArtsVector;
    using MatrixType = ArtsMatrix;
    using ResultType = ArtsMatrix;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    ArtsMatrix() : Matrix() {}

    template <typename T, typename = disable_if<is_same<decay<T>, ArtsMatrix>>>
    ArtsMatrix(const T& t) : Matrix(t) {}

    ArtsMatrix (const ArtsMatrix &A)
        : Matrix(A)
    {
        // Nothing to do here.
    }

    ArtsMatrix(ArtsMatrix &&A)
    {
        this->mrr = A.mrr;
        this->mcr = A.mcr;
        this->mdata  = A.mdata;
        A.mdata = nullptr;
    }

    ArtsMatrix & operator=(const ArtsMatrix &A)
    {
        this->Matrix::operator=(A);
    }

    ArtsMatrix & operator=(ArtsMatrix &&A)
    {
        delete[] this->mdata;
        this->mcr  = A.mcr;
        this->mrr  = A.mrr;
        this->mdata   = A.mdata;
        A.mdata = nullptr;
    }

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    Index rows() {return this->nrows();}
    Index cols() {return this->ncols();}

    RealType & operator()(Index i, Index j)
    {
        return this->get(i,j);
    }

    RealType operator()(Index i, Index j) const
    {
        return this->get(i,j);
    }

    // ------------ //
    //  Arithmetic  //
    // ------------ //

    void accumulate(const ArtsMatrix& B)
    {
        this->operator+=(B);
    }

    void subtract(const ArtsMatrix& B)
    {
        this->operator-=(B);
    }

    ArtsMatrix multiply(const ArtsMatrix &B) const
    {
        ArtsMatrix C; C.resize(this->nrows(), B.ncols());
        ::mult(C, *this, B);
        return C;
    }

    ArtsVector multiply(const ArtsVector &v) const
    {
        ArtsVector w; w.resize(this->nrows());
        ::mult(w, *this, v);
        return w;
    }

    ArtsMatrix transpose_multiply(const ArtsMatrix &B) const
    {
        ArtsMatrix C; C.resize(this->ncols(), B.ncols());
        ::mult(C, ::transpose(*this), B);
        return C;
    }

    ArtsVector transpose_multiply(const ArtsVector &v) const
    {
        ArtsVector w; w.resize(this->ncols());
        ::mult(w, ::transpose(*this), v);
        return w;
    }

    VectorType solve(const VectorType& v) const
    {
        VectorType w; w.resize(this->nrows());
        ::solve(w, *this, v);
        return w;
    }

    ArtsMatrix invert() const
    {
        ArtsMatrix B; B.resize(this->nrows(), this->ncols());
        ::inv(B, *this);
        return B;
    }
    void scale(Numeric c)
    {
        this->operator*=(c);
    }

    ArtsMatrix transpose() const
    {
        ArtsMatrix B = ::transpose(*this);
        return B;
    }

};

/** \brief Arts dense matrix interace wrapper.
 *
 * Simple wrapper class providing an interface to the ARTS matrix class.
 *
 */
class ArtsSparse : public Sparse
{
public:

    using RealType = Numeric;
    using VectorType = ArtsVector;
    using MatrixType = ArtsMatrix;
    using ResultType = ArtsMatrix;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    ArtsSparse() = delete;
    ArtsSparse(const Sparse& A_) : A(A_) {}

    ArtsSparse(const ArtsSparse&) = delete;
    ArtsSparse(ArtsSparse&&)      = delete;

    ArtsSparse & operator=(const ArtsSparse&) = delete;
    ArtsSparse & operator=(ArtsSparse &&)     = delete;

    // ----------------- //
    //   Manipulations   //
    // ----------------- //

    Index rows() {return this->nrows();}
    Index cols() {return this->ncols();}

    // ------------ //
    //  Arithmetic  //
    // ------------ //

    ArtsMatrix multiply(const ArtsMatrix &B) const
    {
        ArtsMatrix C; C.resize(this->nrows(), B.ncols());
        ::mult(C, A, B);
        return C;
    }

    ArtsVector multiply(const ArtsVector &v) const
    {
        ArtsVector w; w.resize(this->nrows());
        ::mult(w, A, v);
        return w;
    }

    operator MatrixType()
    {
        return A;
    }

private:

    const Sparse& A;

};

Numeric dot(const ArtsVector& v, const ArtsVector& w)
{
    return v * w;
}

#endif // INTERFACES_ARTS_WRAPPER_H
