#ifndef MPI_LOG_H
#define MPI_LOG_H

#include "../log.h"

namespace invlib
{

template
<
LogType type
>
class MPILog
{

public:

    MPILog(unsigned int v) : verbosity(v)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    template <typename... Params>
    void init(Params... params) {}

    template <typename... Params>
    void step(Params... params) {}

    template <typename... Params>
    void finalize(Params... params) {}

    template <typename... Params>
    void time(Params... params) {}

private:

    int verbosity;
    int rank;

};

template<>
template<typename... Params>
void MPILog<LogType::SOL_CG>::init(Params... params)
{
    auto tuple = std::make_tuple(params...);
    if (verbosity >= 1 && rank == 0)
    {
        std::cout << std::endl;
        std::cout << "CG Solver:" << std::endl;
        std::cout << "\tTolerance:             " << std::get<0>(tuple) << std::endl;
        std::cout << "\tInitial Residual Norm: " << std::get<1>(tuple) << std::endl;
        std::cout << "\tRight-hand side Norm:  " << std::get<2>(tuple) << std::endl;
    }
}

template<>
template<typename... Params>
void MPILog<LogType::SOL_CG>::step(Params... params)
{
    if (verbosity >= 1 && rank == 0)
    {
        auto tuple = std::make_tuple(params...);
        std::cout<< "Step " << std::setw(5) << std::get<0>(tuple) << ", ";
        std::cout<< "Normalized Residual: " << std::get<1>(tuple) << std::endl;
    }
}

template<>
template<typename... Params>
void MPILog<LogType::SOL_CG>::finalize(Params... params)
{
    if (verbosity >= 1 && rank == 0)
    {
        auto tuple = std::make_tuple(params...);
        std::cout << "Conjugate Gradient method converged after ";
        std::cout << std::get<0>(tuple) << " steps." << std::endl << std::endl;
    }
}


}      // invlib
#endif // MPI_LOG_H
