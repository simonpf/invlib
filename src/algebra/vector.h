#ifndef ALGEBRA_VECTOR
#define ALGEBRA_VECTOR

#include "binary_operation.h"

#include <utility>

template
<
typename Base
>
class Vector : public Base
{
public:

    struct VECTOR_ADDITION_NOT_SUPPORTED {};
    struct VECTOR_MULTIPLICATION_NOT_SUPPORTED {};

    template <typename T>
    using Sum = VECTOR_ADDITION_NOT_SUPPORTED;

    template <typename T>
    using Product = VECTOR_MULTIPLICATION_NOT_SUPPORTED;

    Vector()
        : Base() {}


    Vector(const Vector& v) = default;
    Vector(Vector&& v)      = default;

    Vector& operator=(const Vector& v) = default;
    Vector& operator=(Vector&& v) = default;

    Vector(const Base& v)
        : Base(v) {}

    Vector(Base&& v)
        : Base(v) {}

};

#endif // ALGEBRA_VECTOR
