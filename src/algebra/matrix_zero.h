#ifndef ALGEBRA_ZERO_MATRIX
#define ALGEBRA_ZERO_MATRIX

template
<
typename Real,
typename Matrix
>
class MatrixZero
{

public:

    using MatrixBase = Matrix;

    // ----------------- //
    //   Constructors    //
    // ----------------- //

    MatrixZero() : {}

    // ------------------ //
    //      Addition      //
    // ------------------ //

    Matrix add(const Matrix& B)
    {
        Matrix C(B);
        return C;
    }

    // ------------------ //
    //   Multiplication   //
    // ------------------ //

    template <typename T>
    T multiply(const T& B) const
    {
        return MatrixZero();
    }

    // ----------------- //
    // Addition Operator //
    // ----------------- //

    template <typename T>
        using Sum = T;

    template<typename T>
    const T& operator+(const T &B)
    {
        return B;
    }

    // ----------------------- //
    // Multiplication Operator //
    // ----------------------- //

    MatrixZero operator*(const T &A) const
    {
        return MatrixZero();
    }

};
#endif // ALGEBRA_ZERO_MATRIX
