#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// Pybind11 exports defined in other files.
void exportGeometry(py::module& m);


PYBIND11_MODULE(_contact_modes, m) {
    m.doc() = "contact_modes C++ plugin"; // optional module docstring

    // Call exports.
    exportGeometry(m);
}