#ifndef BOOST_TEST_MODULE
#define BOOST_TEST_MODULE "Backends"
#endif

#include <boost/test/included/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mpl/insert_range.hpp>

#include "invlib/archetypes/vector_archetype.h"
#include "../utility.h"

// Conditional includes.

#ifdef CUDA
#include "invlib/cuda/cuda_sparse.h"
#endif
#ifdef MKL
#include "invlib/mkl/mkl_sparse.h"
#endif

using namespace invlib;

template
<
    typename BackendType
>
void backend_test()
{
    using RealType   = typename BackendType::RealType;
    using MatrixType = Matrix<BackendType>;
    using VectorType = Vector<typename MatrixType::VectorType>;
    using MatrixReferenceType = Matrix<MatrixArchetype<RealType>>;
    using VectorReferenceType = Vector<VectorArchetype<RealType>>;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_m(1, 10);
    std::uniform_int_distribution<> dis_n(1, 10);

    size_t m = dis_m(gen);
    size_t n = dis_n(gen);

    VectorType a = VectorData<RealType>::random(n);
    VectorType b = VectorData<RealType>::random(n);

    VectorReferenceType a_ref(static_cast<VectorData<RealType>>(a));
    VectorReferenceType b_ref(static_cast<VectorData<RealType>>(b));

    // Vector addition.

    a.accumulate(b);
    a_ref.accumulate(b_ref);

    VectorData<RealType> a_base(a);
    VectorData<RealType> a_ref_base(a_ref);

    auto error = maximum_error(a_base, a_ref_base);
    BOOST_TEST((error < 1e-8), "Error: accumulate, maximum difference = "
               << error);

    // Vector difference.

    a.subtract(b);
    a_ref.subtract(b_ref);

    a_base     = a;
    a_ref_base = a_ref;

    error = maximum_error<VectorData<RealType>>(a_base, a_ref_base);
    BOOST_TEST((error < 1e-8), "Error: subtract, maximum difference = "
               << error);

    // Addition of a scalar.

    a.accumulate(1.0);
    a_ref.accumulate(1.0);

    a_base     = a;
    a_ref_base = a_ref;

    error = maximum_error<VectorData<RealType>>(a_base, a_ref_base);
    BOOST_TEST((error < 1e-8), "Error: accumulate scalar, maximum difference = "
               << error);

    // Scaling of a vector.

    a.scale(3.142);
    a_ref.scale(3.142);

    a_base     = a;
    a_ref_base = a_ref;

    error = maximum_error<VectorData<RealType>>(a_base, a_ref_base);
    BOOST_TEST((error < 1e-8), "Error: scale, maximum difference = "
               << error);

    // Dot product.

    RealType d = dot(a,b);
    RealType d_ref = dot(a_ref, b_ref);

    error = std::abs(d - d_ref) / std::max(std::abs(d), std::abs(d_ref));
    BOOST_TEST((error < 1e-8), "Error: dot product, maximum difference = "
               << error);

    // Norm.

    RealType norm = b.norm();
    RealType norm_ref = b_ref.norm();

    error = std::abs(norm - norm_ref) / std::max(std::abs(d), std::abs(d_ref));
    BOOST_TEST((error < 1e-8), "Error: norm, maximum difference = "
               << error);

    // Matrix Vector Product.

    MatrixReferenceType A_ref   = SparseData<RealType>::random(m, n);
    SparseData<RealType> A_base(A_ref);
    MatrixType A(A_base);

    VectorType c = A * a;
    VectorReferenceType c_ref = A_ref * a_ref;

    VectorData<RealType> c_base(c);
    VectorData<RealType> c_ref_base(c_ref);
    error = maximum_error(c_base, c_ref_base);

    BOOST_TEST((error < 1e-8), "Error: matrix vector product, maximum difference = "
               << error);

    // Transposed Matrix Vector Product.

    a     = transp(A) * c;
    a_ref = transp(A_ref) * c_ref;

    c_base     = a;
    c_ref_base = a_ref;

    error = maximum_error(c_base, c_ref_base);
    BOOST_TEST((error < 1e-8), "Error: transposed matrix vector product,"
                               " maximum difference = " << error);
}

#ifdef CUDA
using CudaSparseCR = CudaSparse<double, Representation::CompressedRows>;
using CudaSparseCC = CudaSparse<double, Representation::CompressedColumns>;
using CudaSparseHY = CudaSparse<double, Representation::Hybrid>;
using cuda_types   = boost::mpl::list<CudaSparseCR, CudaSparseCC, CudaSparseHY>;
#else
using cuda_types   = boost::mpl::list<>;
#endif

#ifdef MKL
using MklSparseCO  = MklSparse<double, Representation::Coordinates>;
using MklSparseCR  = MklSparse<double, Representation::CompressedRows>;
using MklSparseCC  = MklSparse<double, Representation::CompressedColumns>;
using MklSparseHY  = MklSparse<double, Representation::Hybrid>;
using mkl_types    = boost::mpl::list<MklSparseCO, MklSparseCR, MklSparseCC, MklSparseHY>;
#else
using mkl_types   = boost::mpl::list<>;
#endif

using matrix_types = typename boost::mpl::insert_range<
    cuda_types,
    typename boost::mpl::end<cuda_types>::type,
    mkl_types>::type;

BOOST_AUTO_TEST_CASE_TEMPLATE(backend, T, matrix_types)
{
    size_t ntests = 1000;
    for (unsigned int i = 0; i < ntests; i++)
    {
        backend_test<T>();
    }
}
