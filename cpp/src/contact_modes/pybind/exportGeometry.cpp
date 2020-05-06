#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <contact_modes/geometry/linear_algebra.hpp>

#include <contact_modes/pybind/incidence_graph_interface.hpp>
#include <contact_modes/pybind/arrangements_interface.hpp>

#include <contact_modes/collision/collide_2d.hpp>
#include <contact_modes/collision/manifold_2d.hpp>


namespace py = pybind11;

void exportGeometry(py::module& m) {
    py::class_<NodePython, NodePythonPtr>(m, "Node")
        .def("rank", &NodePython::rank)
        .def("interior_point", &NodePython::interior_point)
        .def("position", &NodePython::position)
        .def("sign_vector", &NodePython::sign_vector)
        .def("subfaces", &NodePython::subfaces)
        .def("superfaces", &NodePython::superfaces);

    py::class_<IncidenceGraphPython, IncidenceGraphPythonPtr>(m, "IncidenceGraph")
        .def("A", &IncidenceGraphPython::A)
        .def("b", &IncidenceGraphPython::b)
        .def("dim", &IncidenceGraphPython::dim)
        .def("num_k_faces", &IncidenceGraphPython::num_k_faces)
        .def("num_incidences", &IncidenceGraphPython::num_incidences)
        .def("num_nodes", &IncidenceGraphPython::num_nodes)
        .def("node", &IncidenceGraphPython::node)
        .def("rank", &IncidenceGraphPython::rank)
        .def("interior_points", &IncidenceGraphPython::interior_points)
        .def("update_sign_vectors", &IncidenceGraphPython::update_sign_vectors)
        .def("positions", &IncidenceGraphPython::positions)
        .def("sign_vectors", &IncidenceGraphPython::sign_vectors);

    m.def("build_arrangement", &build_arrangement);

    // m.def("initial_arrangement", &initial_arrangement);

    m.def("image_and_kernel_bases", &image_and_kernel_bases);
    m.def("kernel_basis", (Eigen::MatrixXd (*)(const Eigen::MatrixXd&, double)) &kernel_basis);
}

void exportCollision(py::module& m) {

    py::class_<Manifold2D, Manifold2DPtr>(m, "Manifold2D")
        .def_readonly("pts_A", &Manifold2D::pts_A)
        .def_readonly("pts_B", &Manifold2D::pts_B)
        .def_readonly("normal", &Manifold2D::normal)
        .def_readonly("dists", &Manifold2D::dists);

    m.def("collide_2d", &collide_2d);
}
