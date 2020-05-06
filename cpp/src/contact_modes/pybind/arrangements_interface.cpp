#include <contact_modes/pybind/arrangements_interface.hpp>
#include <iostream>


IncidenceGraphPythonPtr 
build_arrangement(Eigen::MatrixXd A, 
                  Eigen::VectorXd b, 
                  double eps) {
    // 
    int d = A.cols();
    int n = A.rows();
    initial_hyperplanes(A, b, eps);
    Eigen::MatrixXd A0 = A.topRows(d);
    Eigen::MatrixXd b0 = b.topRows(d);
    IncidenceGraph* graph = initial_arrangement(A0, b0, eps);
    for (int i = d; i < n; i++) {
        // std::cout << A.rows() << std::endl;
        increment_arrangement(A.row(i), b[i], graph, eps);
    }
    IncidenceGraphPythonPtr graph_py = 
        std::make_shared<IncidenceGraphPython>(graph);
    return graph_py;
}