#ifndef _INVLIB_UTILITY_TUPLE_H_
#define _INVLIB_UTILITY_TUPLE_H_

#include <tuple>

namespace invlib::tuple {

    // --------------------- //
    //  Tuple concatenation  //
    // --------------------- //

    template<typename ... Ts> struct ConcatTuple;

    template <typename ... Ts, typename ... Us>
    struct ConcatTuple<std::tuple<Ts ...>, std::tuple<Us ...>>
    {
        using Type = std::tuple<Ts ..., Us ...>;
    };

    // ----------- //
    //  Tuple map  //
    // ----------- //

    template<template<typename> typename TT, typename ... Ts> struct Map;

    template<template<typename> typename TT, typename T, typename ... Ts>
    struct Map<TT, std::tuple<T, Ts ...>> {
        using RecursiveType = typename Map<TT, std::tuple<Ts ...>>::Type;
        using Type = typename ConcatTuple<std::tuple<TT<T>>, RecursiveType>::Type;
    };

    template<template<typename> typename TT, typename T>
    struct Map<TT, std::tuple<T>> {
        using Type = std::tuple<TT<T>>;
    };
}

#endif
