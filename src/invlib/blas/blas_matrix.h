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

template <typename SType, template <typename> typename VData> class BlasVector;

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
    template <typename> typename MData
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
        axpy(n * m, 1.0, B.elements, 1, elements, 1);
    }

    void subtract(const BlasMatrix &B) {
        axpy(n * m, -1.0, B.elements, 1, elements, 1);
    }

    BlasMatrix multiply(const BlasMatrix &B) const {
        BlasMatrix C; C.resize(m, B.n);
        constexpr char trans = 'n';
        blas::gemm<SType>('n', 'n', B.n, m, n,
                          1.0, B.elements, B.n,
                          elements, n, 0.0,
                          C.elements, C.n);
        return C;
    }

    VectorType multiply(const VectorType &u) const {
        VectorType v; v.resize(m);
        const SType *u_ptr = u.get_element_pointer();
              SType *v_ptr = v.get_element_pointer();
        constexpr char trans = 't';
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n - 1; ++ j) {
            }
        }
        blas::gemv<SType>('t', n, m, 1.0, elements, n, u_ptr, 1, 0.0, v_ptr, 1);
        for (size_t i = 0; i < u.rows(); ++i) {
        }
        return v;
    }

    VectorType transpose_multiply(const VectorType &u) const {
        VectorType v; v.resize(n);
        const SType *u_ptr = u.get_element_pointer();
              SType *v_ptr = v.get_element_pointer();
        constexpr char trans = 'n';
        blas::gemv<SType>('n', m, n, 1.0, elements, m, u_ptr, 1, 0.0, v_ptr, 1);
        return v;
    }

protected:

    // ------------------- //
    //  Base Class Members //
    // ------------------- //

    using MData<SType>::elements;
    using MData<SType>::m;
    using MData<SType>::n;

};

}      // namespace invlib
#endif // BLAS_BLAS_MATRIX_H
