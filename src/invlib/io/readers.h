#ifndef IO_READERS_H
#define IO_READERS_H

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "pugixml/pugixml.hpp"
#include "pugixml/pugixml.cpp"
#include "endian.h"

#include "invlib/types/sparse/sparse_data.h"
#include "invlib/types/dense/vector_data.h"

namespace invlib
{

template<typename Real, typename Index>
using SparseMatrix = SparseData<Real, Index, Representation::Coordinates>;

template <typename Real = double, typename Index = int>
SparseMatrix<Real, Index> read_matrix_arts(const std::string & filename)
{
    // Read xml file.
    pugi::xml_document doc;
    doc.load_file(filename.c_str());
    pugi::xml_node node = doc.document_element();

    // Binary or text format?
    std::string format = node.attribute("format").value();

    // Rows and columns.
    pugi::xml_node sparse_node    = node.child("Sparse");
    std::string rows_string = sparse_node.attribute("nrows").value();
    std::string columns_string = sparse_node.attribute("ncols").value();
    Index m = std::stoi(rows_string);
    Index n = std::stoi(columns_string);

    pugi::xml_node rind_node       = sparse_node.child("RowIndex");
    pugi::xml_node cind_node       = sparse_node.child("ColIndex");
    pugi::xml_node data_node       = sparse_node.child("SparseData");

    std::string nelem_string = rind_node.attribute("nelem").value();

    size_t nelem = static_cast<size_t>(std::stoi(nelem_string));
    std::vector<Index> row_indices(nelem), column_indices(nelem);
    std::vector<Real> elements(nelem);

    if (format == "ascii")
    {
        std::stringstream rind_stream(rind_node.child_value());
        std::stringstream cind_stream(cind_node.child_value());
        std::stringstream data_stream(data_node.child_value());

        std::string cind_string, rind_string, data_string;
        Index rind, cind;
        Real data;

        for (size_t i = 0; i < nelem; i++)
        {
            cind_stream >> cind;
            rind_stream >> rind;
            data_stream >> data;

            row_indices[i]    = rind;
            column_indices[i] = cind;
            elements[i]       = data;
        }
    }
    else if (format == "binary")
    {
        union {
            unsigned char buf[8];
            uint32_t      four;
            uint64_t      eight;
        } buf;

        unsigned int rind, cind;
        double data;

        std::ifstream stream(filename + ".bin", std::ios::in | std::ios::binary);

        if (stream.is_open())
        {
            // Read indices byte by byte and convert from little endian format.
            for (size_t i = 0; i < nelem; i++)
            {
                buf.buf[0] = stream.get();
                buf.buf[1] = stream.get();
                buf.buf[2] = stream.get();
                buf.buf[3] = stream.get();

                rind = le32toh(buf.four);
                row_indices[i] = static_cast<Index>(rind);
            }
            // Read indices byte by byte and convert from little endian format.
            for (size_t i = 0; i < nelem; i++)
            {
                buf.buf[0] = stream.get();
                buf.buf[1] = stream.get();
                buf.buf[2] = stream.get();
                buf.buf[3] = stream.get();
                cind = le32toh(buf.four);
                column_indices[i] = static_cast<Index>(cind);
            }
            // Read data byte by byte and convert from little endian format.
            for (size_t i = 0; i < nelem; i++)
            {
                buf.buf[0] = stream.get();
                buf.buf[1] = stream.get();
                buf.buf[2] = stream.get();
                buf.buf[3] = stream.get();
                buf.buf[4] = stream.get();
                buf.buf[5] = stream.get();
                buf.buf[6] = stream.get();
                buf.buf[7] = stream.get();

                uint64_t host_endian = le64toh(buf.eight);
                data = *reinterpret_cast<double *>(&host_endian);
                elements[i] = static_cast<Real>(data);
            }
        }
    }

    SparseMatrix<Real, Index> matrix(m, n);
    matrix.set(row_indices, column_indices, elements);
    return matrix;
}

template<typename Real = double>
VectorData<Real> read_vector_arts(const std::string & filename)
{
    // Read xml file.
    pugi::xml_document doc;
    doc.load_file(filename.c_str());
    pugi::xml_node node = doc.document_element();

    // Binary or text format?
    std::string format = node.attribute("format").value();

    // Number of elements.
    pugi::xml_node vector_node = node.child("Vector");
    std::string nelem_string   = vector_node.attribute("nelem").value();

    size_t nelem = std::stoi(nelem_string);

    VectorData<Real> v; v.resize(nelem);
    Real * elements = v.get_element_pointer();

    if (format == "ascii")
    {
        std::stringstream elem_stream(vector_node.child_value());
        Real data;

        for (size_t i = 0; i < nelem; i++)
        {
            elem_stream >> data;
            elements[i] = data;
        }
    }
    else if (format == "binary")
    {
        union {
            unsigned char buf[8];
            uint64_t      eight;
        } buf;

        double data;

        std::ifstream stream(filename + ".bin", std::ios::in | std::ios::binary);

        if (stream.is_open())
        {
            // Read data byte by byte and convert from little endian format.
            for (size_t i = 0; i < nelem; i++)
            {
                buf.buf[0] = stream.get();
                buf.buf[1] = stream.get();
                buf.buf[2] = stream.get();
                buf.buf[3] = stream.get();
                buf.buf[4] = stream.get();
                buf.buf[5] = stream.get();
                buf.buf[6] = stream.get();
                buf.buf[7] = stream.get();

                uint64_t host_endian = le64toh(buf.eight);
                data = *reinterpret_cast<double*>(&host_endian);
                elements[i] = static_cast<Real>(data);
            }
        }
    }

    return v;
}

} // namespace invlib

#endif

