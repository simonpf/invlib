#ifndef IO_WRITERS_H
#define IO_WRITERS_H

#include <fstream>
#include <iostream>
#include <string>

#include "pugixml.hpp"
#include "pugixml.cpp"
#include "endian.h"

namespace invlib
{

enum class Format {ASCII, Binary};

template <typename T> void write_matrix_arts(const std::string &,
                                             const T &,
                                             Format format = Format::ASCII);

template <>
void write_matrix_arts(const std::string & filename,
                       const EigenSparse & matrix,
                       Format format)
{
    size_t nnz = matrix.nonZeros();

    std::vector<size_t> rinds{}, cinds{};
    std::vector<double> data{};
    rinds.reserve(matrix.cols());
    cinds.reserve(nnz);
    data.reserve(nnz);

    size_t elemIndex = 0;
    size_t outerIndex = 0;
    size_t innerIndex = 0;
    for (size_t i = 0; i < nnz; i++)
    {
        data.push_back(matrix.valuePtr()[i]);
        rinds.push_back(matrix.innerIndexPtr()[i]);
    }
    for (size_t i = 0; i < matrix.cols(); i++)
    {
        cinds.push_back(matrix.outerIndexPtr()[i]);
    }

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
    ss.str(""); ss << matrix.rows();
    xml_matrix.append_attribute("nrows") = ss.str().c_str();

    auto xml_rind= xml_matrix.append_child("RowIndex");
    ss.str(""); ss << matrix.nonZeros();
    xml_rind.append_attribute("nelem") = ss.str().c_str();


    auto xml_cind= xml_matrix.append_child("ColIndex");
    ss.str(""); ss << matrix.nonZeros();
    xml_cind.append_attribute("nelem") = ss.str().c_str();

    auto xml_data= xml_matrix.append_child("SparseData");
    ss.str(""); ss << matrix.nonZeros();
    xml_cind.append_attribute("nelem") = ss.str().c_str();

    if (format == Format::ASCII)
    {
        ss.str("");
        for (size_t i = 0; i < rinds.size(); i++)
        {
            ss << rinds[i] << " ";
        }
        auto nodechild = xml_rind.append_child(pugi::node_pcdata);
        nodechild.set_value(ss.str().c_str());

        ss.str("");
        for (size_t i = 0; i < cinds.size(); i++)
        {
            ss << data[i] << " ";
        }
        nodechild = xml_data.append_child(pugi::node_pcdata);
        nodechild.set_value(ss.str().c_str());

        ss.str("");
        for (size_t i = 0; i < cinds.size(); i++)
        {
            ss << cinds[i] << " ";
        }
        nodechild = xml_cind.append_child(pugi::node_pcdata);
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
            buf.four = htole32(rinds[i]);
            file.write(buf.buf, 4);
        }
        for (size_t i = 0; i < nnz; i++)
        {
            buf.four = htole32(cinds[i]);
            file.write(buf.buf, 4);
        }
        for (size_t i = 0; i < nnz; i++)
        {
            buf.eight = htole64(cinds[i]);
            file.write(buf.buf, 8);
        }
        file.close();
    }

    xml_doc.save_file(filename.c_str());
}

template <typename T> void write_vector_arts(const std::string &,
                                             const T &,
                                             Format format = Format::ASCII);

template <>
void write_vector_arts(const std::string & filename,
                       const EigenVector & vector,
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

    if (format == Format::ASCII)
    {
        ss.str("");
        for (size_t i = 0; i < nelem; i++)
        {
            ss << vector(i) << " ";
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
            buf.eight = htole64(vector(i));
            file.write(buf.buf, 8);
        }
        file.close();
    }

    xml_doc.save_file(filename.c_str());
}

}      // namespace invlib
#endif // IO_WRITERS_H
