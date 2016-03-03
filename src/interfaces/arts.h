#include arts.h
#include matpack.h

class ArtsMatrix : public Matrix
{

    using Real = Numeric;
    using Base = Matrix

    template <typename ...Args>
    EigenMatrixWrapper( Args &&... params)
        : Base(std::forward<Args>(params)...) {}


    Index cols() {this->ncols()};
    Index rows() {this->nrows()};

    Matrix operator*(const Matrix& B)
    {
        Matrix C; C.resize(A.rows(), B.cols());
        mult(C, A, B);
        return C;
    }

    ArtsMatrix operator*(Numeric c)
    {
        Matrix C; resize
        C*=c;
        return C;
    }

    ArtsMatrix operator+(const Matrix& B)
    {
        ArtsMatrix C = *this;
        C += B;
        return C;
    }

    ArtsMatrix invert() const
    {
        ArtsMatrix B(this->cols(), this->rows());
        inv(B, A);
        return B;
    }

    ArtsMatrix solve(const Vector &v) const
    {
        const Vector w;
        solve(w, *this, v);
        return w;
    }

    ArtsMatrix transpose_add(const ArtsMatrix &B)
    {
        ArtsMatrix C(this->rows(), B.cols());
        mult(C, transpose(*this), B);
    }

    ArtsMatrix transpose_multiply(const ArtsMatrix &B)
    {
        ArtsMatrix C(this->rows(), B.cols());
        mult(C, transpose(*this), B);
    }

    Vector transpose_multiply(const Vector &v)
    {
        Vector w(this->rows());
        mult(w, transpose(*this), v);
        return w;
    }

}
