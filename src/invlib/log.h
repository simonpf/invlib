#ifndef LOG_H
#define LOG_H

#include <tuple>
#include <string>

namespace invlib
{

// ------------------------ //
//    Forward Declarations  //
// ------------------------ //

template
<
typename RealType,
typename Solver
>
class GaussNewton;

template
<
typename RealType,
typename DampingMatrix,
typename Solver
>
class LevenbergMarquardt;

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

// ---------------------- //
//  Formatting Functions  //
// ---------------------- //

std::string center(const std::string &s, int width = 80)
{
    int padding_length = (width - s.size()) / 2;
    std::string padding(padding_length, ' ');
    std::string centered(padding);
    centered += s;
    centered += padding;
    return centered;
}

std::string separator(int width = 80)
{
    std::string separator(width, '-');
    return separator;
}
// ------------------- //
//      Type Names     //
// ------------------- //

template<typename T>
std::string name();

template
<
typename RealType,
typename DampingMatrix,
typename Solver
>
std::string name()
{
    return std::string("Levenberg-Marquardt");
}

template
<
typename RealType,
typename Solver
>
std::string name()
{
    return std::string("Gauss-Newton");
}

// ---------------------- //
//     Log Functions      //
// ---------------------- //

template<>
template<typename... Params>
void StandardLog<LogType::MAP>::init(Params... params)
{
    auto tuple = std::make_tuple(params...);
    if (verbosity >= 1)
    {
        std::cout << center("MAP Computation") << std::endl;

        // Print formulation.
        int formulation = static_cast<int>(std::get<0>(tuple));
        switch (formulation)
        {
        case 0:
                std::cout << "Formulation: Standard" << std::endl;
                break;
        case 1:
                std::cout << "Formulation: N-Form" << std::endl;
                break;

        case 2:
                std::cout << "Formulation: M-Form" << std::endl;
                break;
        }

        // Print optimization method.
        using OptimizationType =
            typename std::tuple_element<1, decltype(tuple)>::type;
        std::cout << "Optimization Method:" << name<OptimizationType>() << std::endl;

        std::cout << separator() << std::endl;
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
