#include <Python.h>
#include "invlib/interfaces/python/python_vector.h"

#define DLL_PUBLIC __attribute__ ((visibility ("default")))

extern "C" {

    enum scalar_type {
        SINGLE = 1,
        DOUBLE = 2
    };

    void* create_vector(void *data,
                        size_t n,
                        bool copy,
                        scalar_type t)
    {
        switch (t) {

        case SINGLE:
        {
            auto v = invlib::PythonVector<float>(reinterpret_cast<float*>(data),
                                                 n,
                                                 copy);
            return &v;
        }

        case DOUBLE:
        {
            auto v = invlib::PythonVector<double>(reinterpret_cast<double*>(data),
                                                  n,
                                                  copy);
            return &v;
        }

        default:
        {
            throw std::runtime_error("Unsupported scalar_type provided.");
        }
        }
    }
}
