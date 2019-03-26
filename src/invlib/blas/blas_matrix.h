/**
 * \file blas/blas_matrix.h
 *
 * \brief Dense matrix arithmetic using BLAS.
 *
 */
#ifndef BLAS_BLAS_MATRIX_H
#define BLAS_BLAS_MATRIX_H

#include "invlib/sparse/sparse_data.h"
#include "invlib/blas/blas_generic.h"

namespace invlib {

// -------------------- //
// Forward Declarations //
// -------------------- //

template <typename SType, template <typename> typename VData> class lasVector;

template <typename Real>
class MatrixData;

// -------------------  //
//   Blas Matrix Class  //
// -------------------  //

/**
 * \brief Dense Matrix Arithmetic using BLAS
 *
 * This class template provides BLAS arithmetic for dense matrices stores
 * contiguously in memory using standard c memory layout.
 *
 * \tparam SType Floating point type used for the representation of
 * the vector elements.
 * \tparam VData Class template that implements the storage of the matrix
 * elements.
 *
 */
template
<
    typename SType,
    template <typename> typename MData = MatrixData
>
class BlasMatrix : public MData<SType>
{
public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType   = SType;
    using VectorType = BlasVector<SType, MData<SType>::template VData>;
    using MatrixType = BlasMatrix<SType, MData>;
    using ResultType = MatrixType;

    friend typename MData<SType>::template VData<SType>;

    template<typename SType2, template <typename> typename VData>
    friend class BlasVector;

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    BlasMatrix()                                = default;
    BlasMatrix(const BlasMatrix &)              = default;
    BlasMatrix(BlasMatrix &&)                   = default;
    BlasMatrix & operator=(const BlasMatrix & ) = default;
    BlasMatrix & operator=(      BlasMatrix &&) = default;

    BlasMatrix(const MData<SType> & m)
    : MData<SType>(m) {
        std::cout << m.rows() << " // " << m.cols() << std::endl;
        // Nothing to do here.
    }

    BlasMatrix(MData<SType> &&m)
    : MData<SType>(std::forward<MData<SType>>(m)) {
        // Nothing to do here.
    }

    // ------------ //
    //  Arithmetic  //
    // ------------ //

    void accumulate(const BlasMatrix &B) {
        axpy(n * m, 1.0, B.get_element_pointer(), 1, get_element_pointer(), 1);
    }

    void subtract(const BlasMatrix &B) {
        axpy(n * m, -1.0, B.get_element_pointer(), 1, get_element_pointer(), 1);
    }

    BlasMatrix multiply(const BlasMatrix &B) const {
        BlasMatrix C; C.resize(m, B.n);
        constexpr char trans = 'n';
        blas::gemm<SType>('n', 'n', B.n, m, n,
                          1.0, B.get_element_pointer(), B.n,
                          get_element_pointer(), n, 0.0,
                          C.get_element_pointer(), C.n);
        return C;
    }

    BlasMatrix transpose_multiply(const BlasMatrix &B) const {
        BlasMatrix C; C.resize(n, B.n);
        constexpr char trans = 'n';
        blas::gemm<SType>('n', 't', B.n, n, m,
                        1.0, B.get_element_pointer(), B.n,
                        get_element_pointer(), n, 0.0,
                        C.get_element_pointer(), C.n);
        return C;
    }

    template<template <typename> typename TT>
    BlasVector<SType, TT> multiply(const BlasVector<SType, TT> &u) const {
        BlasVector<SType, TT> v; v.resize(m);
        const SType *u_ptr = u.get_element_pointer();
              SType *v_ptr = v.get_element_pointer();
        constexpr char trans = 't';
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n - 1; ++ j) {
            }
        }
        blas::gemv<SType>('t', n, m, 1.0, get_element_pointer(), n, u_ptr, 1, 0.0, v_ptr, 1);
        for (size_t i = 0; i < u.rows(); ++i) {
        }
        return v;
    }

    template<template <typename> typename TT>
    BlasVector<SType, TT> transpose_multiply(const BlasVector<SType, TT> &u) const {
        BlasVector<SType, TT> v; v.resize(n);
        const SType *u_ptr = u.get_element_pointer();
              SType *v_ptr = v.get_element_pointer();
        constexpr char trans = 'n';
        blas::gemv<SType>('n', n, m, 1.0, get_element_pointer(), n, u_ptr, 1, 0.0, v_ptr, 1);
        return v;
    }

    using MData<SType>::get_element_pointer;

protected:

    // ------------------- //
    //  Base Class Members //
    // ------------------- //

    using MData<SType>::m;
    using MData<SType>::n;

};

}      // namespace invlib
#endif // BLAS_BLAS_MATRIX_H
