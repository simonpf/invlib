/** \file utility/reference_wrapper.h
 *
 * \brief Wrapper for assignable references.
 *
 */

#ifndef UTILITY_REFERENCE_WRAPPER_H
#define UTILITY_REFERENCE_WRAPPER_H

namespace invlib
{

template
<
typename T1
>
class ReferenceWrapper
{
    using Decayed = typename std::decay<T>::type;

    ReferenceWrapper(T1 &t) : reference_pointer(&t) {}

    ReferenceWrapper & operator=(T1 &t)
    {
        reference_pointer = &t;
    }

    operator T1&()
    {
        return *reference_pointer;
    }

}

}

#endif
