#ifndef _INVLIB_UTILITY_TUPLE_H_
#define _INVLIB_UTILITY_TUPLE_H_

#include <tuple>

namespace invlib::tuple {

    // --------------------- //
    //  Tuple concatenation  //
    // --------------------- //

    template<typename ... Ts> struct Concatenate;

    template <typename ... Ts>
        struct Concatenate<std::tuple<Ts ...>>
    {
        using Type = std::tuple<Ts ...>;
    };

    template <typename ... Ts, typename ... Us>
    struct Concatenate<std::tuple<Ts ...>, std::tuple<Us ...>>
    {
        using Type = std::tuple<Ts ..., Us ...>;
    };

    template <typename ... Ts, typename ... Us, typename ... Vs>
    struct Concatenate<std::tuple<Ts ...>, std::tuple<Us ...>, Vs ...>
    {
        using Recursive = typename Concatenate<std::tuple<Ts ...>, std::tuple<Us ...>>::Type;
        using Type = typename Concatenate<Recursive, Vs ...>::Type;
    };

    // ----------- //
    //  Tuple map  //
    // ----------- //

    template<template<typename> typename TT, typename ... Ts> struct Map;

    template<template<typename> typename TT, typename T, typename ... Ts>
    struct Map<TT, std::tuple<T, Ts ...>> {
        using RecursiveType = typename Map<TT, std::tuple<Ts ...>>::Type;
        using Type = typename Concatenate<std::tuple<TT<T>>, RecursiveType>::Type;
    };

    template<template<typename> typename TT, typename T>
    struct Map<TT, std::tuple<T>> {
        using Type = std::tuple<TT<T>>;
    };
}

#endif
