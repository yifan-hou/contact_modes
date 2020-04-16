#pragma once
#include <contact_modes/geometry/incidence_graph.hpp>


IncidenceGraph* initial_arrangement(const Eigen::MatrixXd& A, 
                                    const Eigen::VectorXd& b, 
                                    double eps);

void increment_arrangement(Eigen::VectorXd a, double b, 
                           IncidenceGraph* I, double eps);