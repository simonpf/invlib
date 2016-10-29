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

template <typename T> void write_matrix_arts(const std::string &, const T &);

template <>
void write_matrix_arts(const std::string & filename, const EigenSparse & matrix)
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
    xml_root.append_attribute("format") = "binary";
    xml_root.append_attribute("version") = "1";

    auto xml_matrix = xml_root.append_child("Sparse");
    ss << matrix.cols();
    xml_matrix.append_attribute("ncols") = ss.str().c_str();
    ss.str(""); ss << matrix.rows();
    xml_matrix.append_attribute("nrows") = ss.str().c_str();

    auto xml_rind= xml_matrix.append_child("RowIndex");
    ss.str(""); ss << matrix.nonZeros();
    xml_rind.append_attribute("nelem") = ss.str().c_str();
    ss.str("");
    for (size_t i = 0; i < rinds.size(); i++)
    {
        ss << rinds[i] << " ";
    }
    auto nodechild = xml_rind.append_child(pugi::node_pcdata);
    nodechild.set_value(ss.str().c_str());

    auto xml_cind= xml_matrix.append_child("ColIndex");
    ss.str(""); ss << matrix.nonZeros();
    xml_cind.append_attribute("nelem") = ss.str().c_str();
    ss.str("");
    for (size_t i = 0; i < cinds.size(); i++)
    {
        ss << cinds[i] << " ";
    }
    nodechild = xml_cind.append_child(pugi::node_pcdata);
    nodechild.set_value(ss.str().c_str());

    auto xml_data= xml_matrix.append_child("SparseData");
    ss.str(""); ss << matrix.nonZeros();
    xml_cind.append_attribute("nelem") = ss.str().c_str();
    ss.str("");
    for (size_t i = 0; i < cinds.size(); i++)
    {
        ss << data[i] << " ";
    }
    nodechild = xml_data.append_child(pugi::node_pcdata);
    nodechild.set_value(ss.str().c_str());

    bool saveSucceeded = xml_doc.save_file(filename.c_str());
    assert(saveSucceeded);
}

template <typename T> void write_vector_arts(const std::string &, const T &);

}

#endif // IO_WRITERS_H
