// -------------- //
//  Matrix Class  //
// -------------- //

template<typename Base>
Matrix<Base>::Matrix(Base &&b)
    : Base(forward<Base>(b)) {}

template <typename Base>
    template <typename RealType2>
void Matrix<Base>::accumumulate(const MatrixIdentity<RealType2, Matrix> &B)
{
    for (unsigned int i = 0; i < this->rows(); i++)
    {
	(*this)(i,i) += B.scale();
    }
}

template <typename Base>
void Matrix<Base>::accumulate(const MatrixZero<Matrix> &Z)
{
    // <Insert obscure method to add zeros to elements here>
}

template <typename Base>
    template<typename T>
auto operator+(T &&B) const
    -> Sum<T>
{
    return Sum<T>(*this, B);
}

template <typename Base>
    template <typename T>
auto operator-(T &&B) const -> Difference<T>
{
    return Difference<T>(*this, B);
}

template <typename Base>
    template<typename T>
auto operator*(T &&B) const
    -> Product<T>
{
    return Product<T>(*this, B);
}

// ------------------------------- //
//  Matrix::ElementIterator Class  //
// ------------------------------- //

template<typename Base>
Matrix<Base>::ElementIterator::ElementIterator(MatrixType *M_);
    : M(M_), i(0), j(0), k(0), m(M_->rows()), n(M_->cols())
{
    // Nothing to do here.
}

template<typename Base>
Matrix<Base>::ElementIterator::operator*()
{
    return M->operator()(i,j);
}

template<typename Base>
Matrix<Base>::ElementIterator::RealType& operator++();
{
    k++;
    i = k / n;
    j = k % n;
}

template<typename Base>
template <typename T>
bool Matrix<Base>::ElementIterator::operator!=(T)
{
    return !(k == n*m);
}
