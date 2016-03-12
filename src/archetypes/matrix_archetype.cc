template
<
typename Real
>
MatrixArchetype::DenseMatrix() {}

template
<
typename Real
>
MatrixArchetype::DenseMatrix(const DenseMatrix &A)
    : m(A.rows()), n(A.cols())
{
    delete data;
    data = new Real[m * n];
    std::copy(std::begin(A.data), std::end(A.data), std::begin(data));
}

template
<
typename Real
>
MatrixArchetype::operator=(const DenseMatrix &A)
    : m(A.rows()), n(A.cols())
{
    delete data;
    data = new Real[m * n];
    std::copy(std::begin(A.data), std::end(A.data), std::begin(data));
}

template
<
typename Real
>
void MatrixArchetype::resize(unsigned int i, unsigned int j)
{
    m = i;
    n = j;
    delete data;
    data = new Real[m * n];
}

template
<
typename Real
>
Real & MatrixArchetype::operator()(unsigned int i, unsigned int j)
{
    assert((0 <= i) && (i < m));
    assert((0 <= j) && (i < n));

    return data[i * n + j];
}

template
<
typename Real
>
Real MatrixArchetype::operator()(unsigned int i, unsigned int j) const
{
    assert((0 <= i) && (i < m));
    assert((0 <= j) && (i < n));

    return data[i * n + j];
}

template
<
typename Real
>
unsigned int MatrixArchetype::cols() const
{
    return n;
}

template
<
typename Real
>
unsigned int MatrixArchetype::rows() const
{
    return m;
}

template
<
typename Real
>
void MatrixArchetype::accumulate(const DenseMatrix &B)
{
    for (unsigned int i = 0; i < m; i++)
    {
	for (unsigned int j = 0; j < n; j++)
	{
	    data[n * i + j] += B(i,j);
	}
    }
}

template
<
typename Real
>
void MatrixArchetype::subtract(const DenseMatrix &B)
{
    for (unsigned int i = 0; i < m; i++)
    {
	for (unsigned int j = 0; j < n; j++)
	{
	    data[n * i + j] -= B(i,j);
	}
    }
}

template
<
typename Real
>
void MatrixArchetype DenseMatrix::multiply(const DenseMatrix &B) const
{
    std::assert(n == B.rows());

    MatrixArchetype C; C.resize(m, B.cols());

    for (unsigned int h = 0; h < m; h++)
    {
	for (unsigned int i = 0; i < B.cols(); i++)
	{
	    Real sum = 0.0;
	    for (unsigned int j = 0; j < n; j++)
	    {
		sum += (*this)(h, j) * B(j, i);
	    }
	    C(h, i) = sum;
	}
    }
    return C;
}

template
<
typename Real
>
void MatrixArchetype DenseMatrix::multiply(const DenseVector &v) const
{
    std::assert(m == v.rows());

    DenseVector w; 
    w.resize(m);

    for (unsigned int i = 0; i < m; i++)
    {
	Real sum = 0.0;
	for (unsigned int j = 0; j < n; j++)
	{
	    sum += (*this)(i, j) * v(i);
	}
	w(i) = sum;
    }
    return w;
}

template
<
typename Real
>
void MatrixArchetype DenseMatrix::scale(Real c)
{
    for (unsigned int i = 0; i < m; i++)
    {
	for (unsigned int j = 0; j < n; j++)
	{
	    (*this)(i, j) *= c;
	}
    }
}
