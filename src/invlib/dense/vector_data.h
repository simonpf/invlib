/**
 * \file archtypes/vector_archetype.h
 *
 * Basic vector data type that serves as prototype for all specific
 * vector types.
 *
 */
#ifndef DENSE_VECTOR_DATA_H
#define DENSE_VECTOR_DATA_H

#include <iostream>
#include <memory>
#include <random>

#include "invlib/invlib.h"
#include "invlib/archetypes/vector_archetype.h"
#include "invlib/utility/functions.h"
#include "invlib/utility/array_deleter.h"

namespace invlib
{

/**
 * \brief General Vector Data Class
 *
 * This class serves as a general representation of vector data and is used
 * as a prototype for all other data types that implement specific arithmetic
 * backends.
 *
 * The data is held in a contiguous array of heap memroy allocated using
 * malloc. The memory is manages by using a shared_ptr to hold the reference
 * to the array. Copying of vectors simply copies the shared_ptr meaning that
 * the copy remains associated to the original.
 *
 * This class does not implement any arithmetic operations but is used only
 * for loading and storing of vectors and conversion between different types.
 *
 * \tparam Real The floating point type used for the representation of vector
 * elements.
 */
template
<
typename Real
>
class VectorData : public Invlib
{

public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType = Real;

    // --------------- //
    //  Static Members //
    // --------------- //

    static VectorData random(size_t n);

    // ------------------------------- //
    //  Constructors and Destructors   //
    // ------------------------------- //

    VectorData()                                 = default;

    /*! Performs a deep copy of the vector data. */
    VectorData(const VectorData & );
    /*! Performs a deep copy of the vector data. */
    VectorData & operator= (const VectorData & );

    VectorData(      VectorData &&)              = default;
    VectorData & operator= (      VectorData &&) = default;

    /*! Create a VectorData object holding n elements from
     *  a shared_ptr holding the reference to the data array
     *
     * \param n The size of the array.
     * \param A shared pointer holding a reference to the array
     * holding the n elements.
     */
    VectorData(size_t n, std::shared_ptr<Real *> elements);
    VectorData(const VectorArchetype<Real> &);

    // ----------------------------- //
    //  Data Access and Manipulation //
    // ----------------------------- //

    const Real * get_element_pointer() const {return *elements;}
          Real * get_element_pointer()       {return *elements;}


          Real * begin()       {return *elements;}
    const Real * begin() const {return *elements;}
          Real * end()         {return *elements + n;}
    const Real * end()   const {return *elements + n;}

    /*! If the vector already holds a reference to a data array, this shared_ptr
     *  is destructed and a new array of the given size is allocated. */
    void resize(size_t n);

    size_t rows() const {return n;}

    /*! Checks the equality of two vectors up to numerical precision.
     *
     * The threshold is set to 1e-4 for single precision floating point numbers and
     * 1e-9 for double precision floating point numbers.
     */
    bool operator==(const VectorData &) const;

    // ------------ //
    //  Conversions //
    // ------------ //

    operator VectorArchetype<Real>() const;

    template<typename T>
    friend std::ostream & operator<<(std::ostream &, const VectorData<T> &);

protected:

    size_t n = 0;
    std::shared_ptr<Real *> elements;

};

/*! Stream vector to string */
template <typename Real>
std::ostream & operator<<(std::ostream &, const VectorData<Real>&);

#include "vector_data.cpp"

}       // namespace invlib
#endif  // DENSE_VECTOR_DATA_H
