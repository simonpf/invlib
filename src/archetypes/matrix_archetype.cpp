template
<
typename Real
>
MatrixArchetype<Real>::MatrixArchetype(const MatrixArchetype<Real> &A)
    : m(A.rows()), n(A.cols())
{
    data = std::unique_ptr<Real[]>(new Real[m * n]);
    std::copy(&A.data[0], &A.data[1], &data[0]);
}

template
<
typename Real
>
MatrixArchetype<Real>& MatrixArchetype<Real>::operator=(const MatrixArchetype &A)
{
    m = A.rows();
    n = A.cols();

    data = std::unique_ptr<Real[]>(new Real[m * n]);
    std::copy(&A.data[0], &A.data[1], &data[0]);
}

template
<
typename Real
>
void MatrixArchetype<Real>::resize(unsigned int i, unsigned int j)
{
    m = i;
    n = j;
    data = std::unique_ptr<Real[]>(new Real[m * n]);
}

template
<
typename Real
>
Real & MatrixArchetype<Real>::operator()(unsigned int i, unsigned int j)
{
    assert((0 <= i) && (i < m));
    assert((0 <= j) && (i < n));

    return data[i * n + j];
}

template
<
typename Real
>
Real MatrixArchetype<Real>::operator()(unsigned int i, unsigned int j) const
{
    assert((0 <= i) && (i < m));
    assert((0 <= j) && (i < n));

    return data[i * n + j];
}

template
<
typename Real
>
unsigned int MatrixArchetype<Real>::cols() const
{
    return n;
}

template
<
typename Real
>
unsigned int MatrixArchetype<Real>::rows() const
{
    return m;
}

template
<
typename Real
>
void MatrixArchetype<Real>::accumulate(const MatrixArchetype &B)
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
void MatrixArchetype<Real>::subtract(const MatrixArchetype &B)
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
auto MatrixArchetype<Real>::multiply(const MatrixArchetype<Real> &B) const
    -> MatrixArchetype
{
    assert(n == B.rows());

    MatrixArchetype<Real> C; C.resize(m, B.cols());

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
template
<
typename Vector    
>
auto MatrixArchetype<Real>::multiply(const Vector &v) const
    -> Vector
{
    assert(m == v.rows());

    Vector w; 
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
void MatrixArchetype<Real>::scale(Real c)
{
    for (unsigned int i = 0; i < m; i++)
    {
	for (unsigned int j = 0; j < n; j++)
	{
	    (*this)(i, j) *= c;
	}
    }
}
