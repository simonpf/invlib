#ifndef LOG_H
#define LOG_H

#include <tuple>

namespace invlib
{

// ------------------ //
//    Standard Log    //
// ------------------ //

enum class LogType {MAP, OPT_GN, OPT_LM, SUB};

template
<
LogType type
>
class StandardLog
{

public:

    StandardLog(unsigned int v) : verbosity(v) {}

    template <typename... Params>
    void init(Params... params) {}

    template <typename... Params>
    void step(Params... params) {}

    template <typename... Params>
    void finalize(Params... params) {}

private:

    int verbosity;
};

template<>
template<typename... Params>
void StandardLog<LogType::MAP>::init(Params... params)
{
    if (verbosity >= 1)
    {
        auto tuple = std::make_tuple(params...);
        std::cout << std::endl;

        bool converged = std::get<0>(tuple);
        if (converged)
        {
            std::cout << "MAP Computation converged." << std::endl;
        }
        else
        {
            std::cout << "MAP Computation NOT converged!" << std::endl;
        }

        std::cout << "\tTotal number of steps: ";
        std::cout << std::get<1>(tuple) << std::endl;
        std::cout << "\tFinal cost function value: ";
        std::cout << std::get<2>(tuple) << std::endl;

    }
}

template<>
template<typename... Params>
void StandardLog<LogType::MAP>::step(Params... params)
{
    if (verbosity >= 2)
    {
        auto tuple = std::make_tuple(params...);
        std::cout<< std::setw(15) << std::get<0>(tuple);
        std::cout<< std::setw(15) << std::get<1>(tuple);
        std::cout<< std::setw(15) << std::get<2>(tuple);
        std::cout<< std::setw(15) << std::get<3>(tuple);
        std::cout << std::endl;
    }
}

template<>
template<typename... Params>
void StandardLog<LogType::MAP>::finalize(Params... params)
{
    if (verbosity >= 1)
    {
        auto tuple = std::make_tuple(params...);
        std::cout << std::endl;

        bool converged = std::get<0>(tuple);
        if (converged)
        {
            std::cout << "MAP Computation converged." << std::endl;
        }
        else
        {
            std::cout << "MAP Computation NOT converged!" << std::endl;
        }

        std::cout << "\tTotal number of steps: ";
        std::cout << std::get<1>(tuple) << std::endl;
        std::cout << "\tFinal cost function value: ";
        std::cout << std::get<2>(tuple) << std::endl;

    }
}

}      // namespace invlib

#endif // LOG_H
