#ifndef BENCHMARKS_UTILITY_H
#define BENCHMARKS_UTILITY_H

// -------------------- //
//  Tuple Concatenation //
// -------------------- //

template<typename ... Ts> struct ConcatTuple;

template <typename ... Ts, typename ... Us>
struct ConcatTuple<std::tuple<Ts ...>, std::tuple<Us ...>>
{
    using Type = std::tuple<Ts ..., Us ...>;
};

// --------------------------- //
//  Vector to Tuple Converter  //
// --------------------------- //

template<typename t, size_t l, size_t start = 0>
struct VectorToTuple
{
    VectorToTuple(const std::vector<t> & v)
    {
        value = std::tuple_cat(std::make_tuple(v[start]),
                               VectorToTuple<t, l, start + 1>(v).value);
    }

    decltype(std::tuple_cat(std::make_tuple(static_cast<double>(1.0)),
                            VectorToTuple<t, l, start + 1>::value)) value;
};

template<typename t, size_t l>
struct VectorToTuple<t, l, l>
{
    VectorToTuple(const std::vector<t> & v)
    {
        value = std::make_tuple();
    }

    std::tuple<> value;
};

template <size_t l, typename T>
auto vector_to_tuple(const std::vector<T> & v)
    -> decltype(VectorToTuple<T, l>(v).value)
{
    return VectorToTuple<T, l>(v).value;
}

// ----------------- //
//   Tuple Printer   //
// ----------------- //

/*! Tuple Printer
 *
 * Helper functor to print a tuple of values. Each value in the tuple is printed
 * to std::out using the << operator. For each element in the tuple the width
 * is set to 15 charaters.
 */
template
<
typename T,
size_t index = 0,
size_t end = std::tuple_size<T>::value - 1
>
struct TuplePrinter
{
    static void print(const T & tuple, char sep = ',')
    {
        std::cout << std::setw(15) << std::get<index>(tuple) << sep;
        TuplePrinter<T, index + 1>::print(tuple, sep);
    }
};

template
<
typename T,
size_t index
>
struct TuplePrinter<T, index, index>
{
    static void print(const T & tuple, char sep = ',')
    {
        std::cout << std::setw(15) << std::get<index>(tuple);
    }
};

// -------------------- //
//  Benchmark Executor  //
// -------------------- //

template
<
template <typename> class BenchmarkFunction,
typename T,
char sep = ','
>
class Benchmark
{
public:

    Benchmark()                               = default;
    Benchmark(const Benchmark & )             = delete;
    Benchmark(      Benchmark &&)             = delete;
    Benchmark & operator=(const Benchmark &)  = delete;
    Benchmark & operator=(      Benchmark &&) = delete;

    void run() {}

};

template
<
template <typename> class BenchmarkFunction,
char sep,
typename T,
typename ... Ts
>
class Benchmark<BenchmarkFunction, std::tuple<T, Ts ...>, sep>
{
public:

    Benchmark()                               = default;
    Benchmark(const Benchmark & )             = delete;
    Benchmark(      Benchmark &&)             = delete;
    Benchmark & operator=(const Benchmark &)  = delete;
    Benchmark & operator=(      Benchmark &&) = delete;

    void run() {

        // Run benchmark for current type.
        auto result = benchmark();

        // Print results.
        std::cout << typeid(T).name() << " " << sep;
        TuplePrinter<decltype(result)>::print(result, sep);
        std::cout << std::endl;

        // Recursion.
        tail.run();
    }

private:

    BenchmarkFunction<T> benchmark;
    Benchmark<BenchmarkFunction, std::tuple<Ts...>, sep> tail;
};


#endif // BENCHMARKS_UTILITY_H
