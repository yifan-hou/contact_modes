#pragma once
#include <contact_modes/geometry/arrangements.hpp>
#include <contact_modes/pybind/incidence_graph_interface.hpp>


IncidenceGraphPythonPtr 
build_arrangement(Eigen::MatrixXd A, 
                  Eigen::VectorXd b, 
                  double eps);