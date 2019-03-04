/**
 * \file archtypes/sparse.h
 *
 * \brief Simple deleter class for shared pointer of array types.
 *
 */
#ifndef UTILITY_ARRAY_H
#define UTILITY_ARRAY_H

namespace invlib{
namespace array{

// ------------------ //
//   Array deleter    //
// ------------------ //

/*! Destructor for shared_ptr of array types.
 *
 * Calls the delete[] destructor for array types created using new[]. This
 * is necessary to properly destroy shared objects of array type.
 *
 * \tparam T The underlying type of the array T[].
 */
template <typename T>
struct Deleter
{
    Deleter(bool owner_ = true) : owner(owner_)
    {
        // Nothing to do here.
    }
    Deleter(const Deleter & ) = default;
    Deleter(      Deleter &&) = default;

    void operator() (const T* ptr)
    {
        if (owner) {
            delete[] ptr;
        }
    }

private:

    bool owner = true;

};

// -------------------- //
//   Create shared_ptr  //
// -------------------- //

/* @brief Convert external array to C++ std::shared_ptr.
 *
 * This function takes a pointer to an array of data and converts it
 * into a shared pointer to the same array.
 *
 * This will be done with or without copying depending on the value
 * of the copy argument. If copy is true, a new array of the given size
 * is created and the data copied. The created shared pointer will then
 * take care of releasing the memory on destruction.
 * If copy is false, no new array is created and the shared pointer will
 * not release the memory upon destruction.
 */
template <typename T>
    std::shared_ptr<T[]> make_shared(void *a, size_t n, bool copy) {
    auto a_ = reinterpret_cast<T *>(a);
    T* ptr  = a_;
    if (copy) {
        ptr = new T[n];
        std::copy(a_, a_ + n, ptr);
    }
    return std::shared_ptr<T[]>(ptr, Deleter<T>(copy));
}

template <typename T>
std::shared_ptr<T[]> create(size_t n) {
    T* array_ptr = new T[n];
    return std::shared_ptr<T[]>(array_ptr, Deleter<T>(true));
}

}      // namespace array
}      // namespace invlib
#endif // UTILITY_ARRAY_H
