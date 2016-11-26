#ifndef IO_WRITERS_H
#define IO_WRITERS_H

#include <fstream>
#include <iostream>
#include <string>

#include "pugixml.hpp"
#include "pugixml.cpp"
#include "endian.h"

#include "invlib/sparse/sparse_data.h"
#include "invlib/dense/vector_data.h"

namespace invlib
{

enum class Format {ASCII, Binary};

using SparseMatrix = SparseData<double, Representation::Coordinates>;

void write_matrix_arts(const std::string & filename,
                       const SparseMatrix & matrix,
                       Format format)
{
    size_t nnz = matrix.non_zeros();

    std::stringstream ss;

    pugi::xml_document xml_doc;
    auto declarationNode = xml_doc.append_child(pugi::node_declaration);
    declarationNode.append_attribute("version")    = "1.0";

    auto xml_root = xml_doc.append_child("arts");

    if (format == Format::Binary)
    {
        xml_root.append_attribute("format") = "binary";
    } else {
        xml_root.append_attribute("format") = "ascii";
    }

    xml_root.append_attribute("version") = "1";

    auto xml_matrix = xml_root.append_child("Sparse");
    ss << matrix.cols();
    xml_matrix.append_attribute("ncols") = ss.str().c_str();
    ss.str(""); ss << nnz;
    xml_matrix.append_attribute("nrows") = ss.str().c_str();

    auto xml_rind= xml_matrix.append_child("RowIndex");
    ss.str(""); ss << matrix.non_zeros();
    xml_rind.append_attribute("nelem") = ss.str().c_str();


    auto xml_cind= xml_matrix.append_child("ColIndex");
    ss.str(""); ss << nnz;
    xml_cind.append_attribute("nelem") = ss.str().c_str();

    auto xml_data= xml_matrix.append_child("SparseData");
    ss.str(""); ss << nnz;
    xml_data.append_attribute("nelem") = ss.str().c_str();

    if (format == Format::ASCII)
    {
        ss.str("");
        for (size_t i = 0; i < nnz; i++)
        {
            ss << matrix.get_row_index_pointer()[i] << " ";
        }
        auto nodechild = xml_rind.append_child(pugi::node_pcdata);
        nodechild.set_value(ss.str().c_str());

        ss.str("");
        for (size_t i = 0; i < nnz; i++)
        {
            ss << matrix.get_column_index_pointer()[i] << " ";
        }
        nodechild = xml_cind.append_child(pugi::node_pcdata);
        nodechild.set_value(ss.str().c_str());

        ss.str("");
        for (size_t i = 0; i < nnz; i++)
        {
            ss << matrix.get_element_pointer()[i] << " ";
        }
        nodechild = xml_data.append_child(pugi::node_pcdata);
        nodechild.set_value(ss.str().c_str());

    }
    else
    {
        std::ofstream file(filename + ".bin", std::ios::out | std::ios::binary);

        union {
            char buf[8];
            uint32_t      four;
            uint64_t      eight;
        } buf;

        for (size_t i = 0; i < nnz; i++)
        {
            buf.four = htole32(matrix.get_row_index_pointer()[i]);
            file.write(buf.buf, 4);
        }
        for (size_t i = 0; i < nnz; i++)
        {
            buf.four = htole32(matrix.get_column_index_pointer()[i]);
            file.write(buf.buf, 4);
        }
        for (size_t i = 0; i < nnz; i++)
        {
            buf.eight = htobe64(reinterpret_cast<const uint64_t *>(matrix.get_element_pointer())[i]);
            file.write(buf.buf, 8);
        }
        file.close();
    }

    xml_doc.save_file(filename.c_str());
}

void write_vector_arts(const std::string & filename,
                       const VectorData<double> & vector,
                       Format format)
{
    size_t nelem = vector.rows();

    std::stringstream ss;

    pugi::xml_document xml_doc;
    auto declarationNode = xml_doc.append_child(pugi::node_declaration);
    declarationNode.append_attribute("version")    = "1.0";

    auto xml_root = xml_doc.append_child("arts");

    if (format == Format::Binary)
    {
        xml_root.append_attribute("format") = "binary";
    } else {
        xml_root.append_attribute("format") = "ascii";
    }

    xml_root.append_attribute("version") = "1";

    auto xml_vector = xml_root.append_child("Vector");
    ss << nelem;
    xml_vector.append_attribute("nelem") = ss.str().c_str();

    const double * elements = vector.get_element_pointer();
    if (format == Format::ASCII)
    {
        ss.str("");
        for (size_t i = 0; i < nelem; i++)
        {
            ss << elements[i] << " ";
        }
        auto nodechild = xml_vector.append_child(pugi::node_pcdata);
        nodechild.set_value(ss.str().c_str());
    }
    else
    {
        std::ofstream file(filename + ".bin", std::ios::out | std::ios::binary);

        union {
            char buf[8];
            uint32_t      four;
            uint64_t      eight;
        } buf;

        for (size_t i = 0; i < nelem; i++)
        {
            buf.eight = htobe64(*reinterpret_cast<const uint64_t *>(elements + i));
            file.write(buf.buf, 8);
        }
        file.close();
    }

    xml_doc.save_file(filename.c_str());
}

}      // namespace invlib
#endif // IO_WRITERS_H
