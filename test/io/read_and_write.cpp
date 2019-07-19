#include "invlib/io/writers.h"
#include "invlib/io/readers.h"
#include "utility.h"
#include "test_types.h"

using namespace invlib;

template <typename T>
struct ReadAndWriteTest {

    static constexpr char name[] = "Linear";

    static void run() {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis_m(1, 1000);
        std::uniform_int_distribution<> dis_n(1, 1000);

        size_t m = 4; dis_m(gen);
        size_t n = 4; dis_n(gen);

        // Generate random sparse matrix.
        using SparseMatrix = SparseData<double, int, Representation::Coordinates>;
        auto A = SparseMatrix::random(m, n);

        // Write and read matrix.

        invlib::write_matrix_arts("test_sparse_binary.xml", A, Format::Binary);
        auto B = invlib::read_matrix_arts("test_sparse_binary.xml");

        ensure_true(A == B, "Writing and reading Arts binary sparse matrix format.");

        invlib::write_matrix_arts("test_sparse_ascii.xml", A, Format::ASCII);
        B = invlib::read_matrix_arts("test_sparse_ascii.xml");


        ensure_true(A == B, "Writing and reading Arts ASCII sparse matrix format.");

        using Vector = VectorData<double>;
        auto v = Vector::random(n);

        // Write and read vector.

        invlib::write_vector_arts("test_vector_binary.xml", v, Format::Binary);
        auto w = invlib::read_vector_arts("test_vector_binary.xml");

        ensure_true(v == w, "Writing and reading Arts binary vector format.");

        invlib::write_vector_arts("test_vector_ascii.xml", v, Format::ASCII);
        w = invlib::read_vector_arts("test_vector_ascii.xml");

        ensure_true(v == w, "Writing and reading Arts ASCII vector format.");
    }
};

TESTMAIN(GenericTest<ReadAndWriteTest COMMA matrix_types>::run();)
