#include "invlib/blas/blas_vector.h"
#include "invlib/blas/blas_matrix.h"
#include "invlib/dense/matrix_data.h"
#include "invlib/mkl/mkl_sparse.h"
#include "invlib/mpi/mpi_matrix.h"
#include "invlib/mpi/mpi_vector.h"
#include "invlib/interfaces/python/general.h"

namespace invlib {

    template
    <
    typename LocalType,
    template <typename> class StorageTrait = LValue
    >
    class PythonMpiWrapper : public MpiMatrix<LocalType, StorageTrait>
    {
    public:

        using Base = MpiMatrix<LocalType, StorageTrait>;
        using RealType  = typename Base::RealType;
        using Base::local;

        PythonMpiWrapper();
        PythonMpiWrapper(const PythonMpiWrapper &) = default;
        PythonMpiWrapper(PythonMpiWrapper &&)      = default;

        template <typename ... Ts>
        PythonMpiWrapper(Ts && ... ts) :
    Base(std::forward<Ts>(ts) ...) {
        // Nothing to do here.
    }


        auto rows() const {
            return local.rows();
        }

        auto cols() const {
            return local.cols();
        }

        auto non_zeros() const {
            return local.non_zeros();
        }

        auto get_index_pointer()  {
            return local.get_index_pointer();
        }

        auto get_index_pointers() {
            return local.get_index_pointers();
        }

        auto get_start_pointer() {
            return local.get_start_pointer();
        }

        auto get_start_pointers() {
            return local.get_start_pointers();
        }

        auto get_element_pointer() {
            return local.get_element_pointer();
        }

        auto get_element_pointers() {
            return local.get_element_pointers();
        }

        PythonMpiWrapper multiply(const PythonMpiWrapper &) const {
            throw std::runtime_error("Multiplication of Python MPI matrices is not supported.");
        }

        template<typename T, typename U = disable_if<is_same<T, PythonMpiWrapper>>>
        auto multiply(const T &t) const -> decltype(Base::multiply(t)) {
            return Base::multiply(t);
        }

        PythonMpiWrapper transpose_multiply(const PythonMpiWrapper &) const {
            throw std::runtime_error("Multiplication of Python MPI matrices is not supported.");
        }

        template<typename T, typename U = disable_if<is_same<T, PythonMpiWrapper>>>
        auto transpose_multiply(const T &t) const -> decltype(Base::multiply(t)) {
            return Base::transpose_multiply(t);
        }
    };


    template<typename ScalarType, typename IndexType>
        struct architecture_trait<ScalarType, IndexType, Architecture::Mpi> {
        using RealType  = ScalarType;
        using Vector    = BlasVector<ScalarType>;

        using DenseData = MatrixData<ScalarType>;
        using Dense = PythonMpiWrapper<BlasMatrix<ScalarType, MatrixData>>;

        using SparseCscData = SparseData<ScalarType, IndexType, Representation::CompressedColumns>;
        using SparseCsrData = SparseData<ScalarType, IndexType, Representation::CompressedRows>;
        using SparseHybData = SparseData<ScalarType, IndexType, Representation::Hybrid>;

        using SparseCsc = PythonMpiWrapper<MklSparse<ScalarType, Representation::CompressedColumns>>;
        using SparseCsr = PythonMpiWrapper<MklSparse<ScalarType, Representation::CompressedRows>>;
        using SparseHyb = PythonMpiWrapper<MklSparse<ScalarType, Representation::Hybrid>>;
    };
}
