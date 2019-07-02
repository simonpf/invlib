#ifndef MPI_GENERIC_H
#define MPI_GENERIC_H

namespace invlib
{

    template<typename T>
    const T & broadcast(const T & t) {
        return t;
    }

}      // namespace invlib
#endif // MPI_MPI_MATRIX_H
