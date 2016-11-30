/**
 * \file archtypes/sparse.h
 *
 * \brief Simple deleter class for shared pointer of array types.
 *
 */
#ifndef UTILITY_ARRAY_DELETER
#define UTILITY_ARRAY_DELETER

namespace invlib
{

/*! Destructor for shared_ptr of array types.
 *
 * Calls the delete[] destructor for array types created using new[]. This
 * is necessary to properly destroy shared objects of array type.
 *
 * \tparam T The underlying type of the array T[].
 */
template <typename T>
struct ArrayDeleter
{
    ArrayDeleter()                      = default;
    ArrayDeleter(const ArrayDeleter & ) = default;
    ArrayDeleter(      ArrayDeleter &&) = default;

    void operator() (const T * ptr)
    {
        delete[] (* ptr);
        delete ptr;
    }
};

}      // namespace invlib
#endif // UTILITY_ARRAY_DELETER
