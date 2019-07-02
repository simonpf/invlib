#ifndef _INTERFACES_PYTHON_GENERAL_H_
#define _INTERFACES_PYTHON_GENERAL_H_

namespace invlib {
    enum class Architecture : unsigned {Cpu, Mpi, Gpu};
    enum class Format       : unsigned {Dense, SparseCsc, SparseCsr, SparseHyb};

    template<typename ScalarType, typename IndexType, Architecture arch> struct architecture_trait;
    template<typename Arch, typename Matrix> struct format_trait;
}
#endif // INTERFACES_PYTHON_PYTHON_MATRIX
