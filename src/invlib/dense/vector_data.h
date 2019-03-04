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
#include "invlib/utility/array.h"

namespace invlib
{

/**
 * \brief General Vector Data Class
 *
 * This class serves as a general representation of vector data and is used
 * as a prototype for all other data types that implement specific arithmetic
 * backends.
 *
 * The data is held in a contiguous array of heap memory allocated using
 * malloc. The memory is managed using a shared_ptr holding the reference
 * to the array. Copying of vectors simply copies the shared_ptr meaning that
 * the copy remains associated to the original.
 *
 * This class does not implement any arithmetic operations but is used only
 * for loading and storing of vectors and conversion between different types.
 *
 * \tparam ScalarType The floating point type used for the representation of vector
 * elements.
 */
template
<
typename ScalarType
>
class VectorData : public Invlib
{

public:

    // -------------- //
    //  Type Aliases  //
    // -------------- //

    using RealType = ScalarType;

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
    ~VectorData() = default;

    /*! Create a VectorData object holding n elements from
     *  a shared_ptr holding the reference to the data array
     *
     * \param n The size of the array.
     * \param A shared pointer holding a reference to the array
     * holding the n elements.
     */
    VectorData(size_t n, std::shared_ptr<ScalarType[]> elements);
    VectorData(const VectorArchetype<ScalarType> &);

    // ----------------------------- //
    //  Data Access and Manipulation //
    // ----------------------------- //

    const ScalarType * get_element_pointer() const {return *elements;}
          ScalarType * get_element_pointer()       {return *elements;}


          ScalarType * begin()       {return elements.get();}
    const ScalarType * begin() const {return elements.get();}
          ScalarType * end()         {return elements.get() + n;}
    const ScalarType * end()   const {return elements.get() + n;}

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

    operator VectorArchetype<ScalarType>() const;

    template<typename T>
    friend std::ostream & operator<<(std::ostream &, const VectorData<T> &);

protected:

    size_t n = 0;
    std::shared_ptr<ScalarType[]> elements;

};

/*! Stream vector to string */
template <typename ScalarType>
std::ostream & operator<<(std::ostream &, const VectorData<ScalarType>&);

#include "vector_data.cpp"

}       // namespace invlib
#endif  // DENSE_VECTOR_DATA_H
