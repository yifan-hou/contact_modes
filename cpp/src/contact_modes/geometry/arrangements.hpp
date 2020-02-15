#pragma once
#include <contact_modes/geometry/incidence_graph.hpp>


IncidenceGraphPtr initial_arrangement(const Eigen::MatrixXd& A, 
                                      const Eigen::VectorXd& b, 
                                      double eps);

void increment_arrangement(const Eigen::VectorXd& a, double b, 
                           IncidenceGraphPtr I, double eps);